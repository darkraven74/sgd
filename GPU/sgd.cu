#include "sgd.cuh"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#define BLOCK_SIZE 4

static unsigned int g_seed;

inline void fast_srand(int seed)
{
    g_seed = seed;
}

inline int fastrand()
{
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        //  Handle error
        return 0;
    }
    return (double) time.tv_sec * 1000000 + (double) time.tv_usec;
}

sgd::sgd(std::istream &tuples_stream,
         int count_features,
         float learning_rate,
         float lambda,
         float alpha,
         int count_samples,
         int likes_format)
    :
    _count_users(0),
    _count_items(0),
    _count_features(count_features),
    _sgd_learning_rate(learning_rate),
    _sgd_lambda(lambda),
    _sgd_alpha(alpha)
{
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

//    srand(time(NULL));
    srand(34);
    fast_srand(34);


    read_likes(tuples_stream, count_samples, likes_format);

    generate_test_set();

    _features_users.assign(_count_users * _count_features, 0);
    _features_items.assign(_count_items * _count_features, 0);
}

sgd::~sgd()
{

}

void sgd::read_likes(std::istream &tuples_stream, int count_simples, int format)
{
    std::string line;
    char const tab_delim = '\t';
    int i = 0;

    while (getline(tuples_stream, line)) {
        std::istringstream line_stream(line);
        std::string value;
        getline(line_stream, value, tab_delim);
        unsigned long uid = atol(value.c_str());

        if (_users_map.find(uid) == _users_map.end()) {
            _users_map[uid] = _count_users;
            _count_users++;
            _user_likes.push_back(std::vector<int>());
            _user_likes_weights.push_back(std::vector<float>());
            _user_likes_weights_temp.push_back(std::vector<float>());
        }

        int user = _users_map[uid];

        if (format == 0) {
            //getline(line_stream, value, tab_delim);
        }

        getline(line_stream, value, tab_delim);
        unsigned long iid = atol(value.c_str());
        float weight = 1;

        float weight_temp = 1;

        if (format == 1) {
            getline(line_stream, value, tab_delim);
            weight_temp = atof(value.c_str());
            //weight = weight_temp;
        }

        if (_items_map.find(iid) == _items_map.end()) {
            _items_map[iid] = _count_items;
            _item_likes.push_back(std::vector<int>());
            _item_likes_weights.push_back(std::vector<float>());
            _count_items++;
        }

        int item = _items_map[iid];
        ///
        /// adding data to user likes
        /// and to item likes
        ///
        _user_likes[user].push_back(item);
        _user_likes_weights[user].push_back(weight);
        _user_likes_weights_temp[user].push_back(weight_temp);
        _item_likes[item].push_back(user);
        _item_likes_weights[item].push_back(weight);

        if (i % 10000 == 0) std::cout << i << " u: " << _count_users << " i: " << _count_items << "\r";

        i++;
        if (count_simples && i >= count_simples) break;
    }

    std::cout.flush();
    std::cout << "\ntotal:\n u: " << _count_users << " i: " << _count_items << std::endl;
}

void sgd::generate_test_set()
{
    int total_size = 0;
    for (int idx = 0; idx < 10000;) {
        //for (int i = 0; i < _count_users; i++) {
        int i = rand() % _count_users;
        if (_user_likes[i].size() < 2) {
            continue;
        }
        idx++;
        total_size += _user_likes[i].size();
        int size = _user_likes[i].size();
        for (int j = 0; j < size / 2;) {
            int id = rand() % _user_likes[i].size();

            /*if (_user_likes_weights_temp[i][id] < 4) {
                continue;
            }*/
            test_set.push_back(std::make_pair(i, _user_likes[i][id]));

            for (unsigned int k = 0; k < _item_likes[_user_likes[i][id]].size(); k++) {
                if (_item_likes[_user_likes[i][id]][k] == i) {
                    _item_likes[_user_likes[i][id]].erase(_item_likes[_user_likes[i][id]].begin() + k);
                    _item_likes_weights[_user_likes[i][id]]
                        .erase(_item_likes_weights[_user_likes[i][id]].begin() + k);
                }
            }

            _user_likes[i].erase(_user_likes[i].begin() + id);
            _user_likes_weights[i].erase(_user_likes_weights[i].begin() + id);
            _user_likes_weights_temp[i].erase(_user_likes_weights_temp[i].begin() + id);
            break;
        }
    }
}

void sgd::fill_rnd(features_vector &in_v, int in_size)
{
    std::cerr << "Generate random features.. ";
    for (int i = 0; i < in_size * _count_features; i++) {
        in_v[i] = ((float) rand() / (float) RAND_MAX);
    }

    std::cerr << "done" << std::endl;
}

void sgd::calculate(int count_iterations, int positive_ratings, int negative_ratings, int is_gpu_calc)
{
    fill_rnd(_features_users, _count_users);
    fill_rnd(_features_items, _count_items);

    std::ofstream hr10("hr10.txt");

    _positive_ratings = positive_ratings;
    _negative_ratings = negative_ratings;

    float old_hr = 0;
    float new_hr = 0.00000001;

    transfers = 0;
    calc = 0;

    std::cout << "is_gpu_calc: " << is_gpu_calc << std::endl;

    for (int i = 0; i < count_iterations; i++) {
        double start = get_wall_time();
        std::cerr << "SGD Iteration: " << i << std::endl;

        if (is_gpu_calc) {
            train_random_preferences_gpu();
        }
        else {
            train_random_preferences_cpu();
        }

        /*std::cout << "users fea: " << std::endl;
        for (int j = 0; j < 7; j++) {
            std::cout << _features_users[j] << " ";
        }
        std::cout << std::endl;

        std::cout << "items fea: " << std::endl;
        for (int j = 0; j < 7; j++) {
            std::cout << _features_items[j] << " ";
        }
        std::cout << std::endl;*/

        cudaDeviceSynchronize();
        double end = get_wall_time();

        std::cerr << "==== Iteration time : " << (end - start) / 1000000 << std::endl;


        old_hr = new_hr;
        new_hr = hit_rate_cpu();
        if (new_hr > old_hr) {
            _sgd_learning_rate *= 1.05;
        }
        else {
            _sgd_learning_rate *= 0.5;
        }
        hr10 << new_hr << std::endl;

    }

    std::cout << "Profiler:\n";
    std::cout << "Transfers: " << transfers / 1000000 << std::endl;
    std::cout << "Calc: " << calc / 1000000 << std::endl;


    hr10.close();

}

__global__ void update_features_2d_gpu(int *item_ids, float *preferences,
                                       float *features_users, float *features_items, int ratings_count,
                                       int _count_features, float _sgd_lambda, float _sgd_learning_rate)
{
    __shared__ float err[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < ratings_count && feature_idx < _count_features) {
        int item_id = item_ids[idx];
        if (item_id != -1) {
            int user_feat_idx = idx * _count_features + feature_idx;
            int item_feat_idx = item_id * _count_features + feature_idx;

            float user_feature = features_users[user_feat_idx];
            float item_feature = features_items[item_feat_idx];

            err[threadIdx.x] = preferences[idx];
            __syncthreads();

            atomicAdd(&err[threadIdx.x], -user_feature * item_feature);

            __syncthreads();

            float error = err[threadIdx.x];

            float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
            float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

            features_users[user_feat_idx] = user_feature + (_sgd_learning_rate * delta_user_feature);
            features_items[item_feat_idx] = item_feature + (_sgd_learning_rate * delta_item_feature);
        }
    }
}

void sgd::train_random_preferences_cpu()
{
    thrust::device_vector<unsigned int> random_data(3 * _count_users);
    std::vector<unsigned int> random_data_host(3 * _count_users);

    int batch_iter_count = 10;
    for (int i = 0; i < batch_iter_count; i++) {
        double start = get_wall_time();

        std::vector<int> is_positive_ratings(_count_users);
        std::vector<int> rand_user_items(_count_users);
        std::vector<int> rand_items(_count_users);

        /*for (int j = 0; j < _count_users; j++) {
            is_positive_ratings[j] = fastrand() % 10;
            rand_user_items[j] = fastrand() % _user_likes[j].size();
            rand_items[j] = fastrand() % _count_items;
        }*/

        curandGenerate(gen, thrust::raw_pointer_cast(&random_data.front()), 3 * _count_users);
        thrust::copy(random_data.begin(), random_data.end(), random_data_host.begin());
        cudaDeviceSynchronize();


#pragma omp parallel for num_threads(omp_get_max_threads())
        for (int j = 0; j < _count_users; j++) {
            is_positive_ratings[j] = random_data_host[j] % 10;
            rand_user_items[j] = random_data_host[j + 1] % _user_likes[j].size();
            rand_items[j] = random_data_host[j + 2] % _count_items;
        }

        std::vector<int> small_item_id(_count_users);
        std::vector<float> small_preference(_count_users);

#pragma omp parallel for num_threads(omp_get_max_threads())
        for (int user = 0; user < _count_users; user++) {
            if (is_positive_ratings[user] < 1) {
                int item_id = rand_user_items[user];
                int item = _user_likes[user][item_id];
                small_item_id[user] = item;
                small_preference[user] = 1 + _sgd_alpha * _user_likes_weights[user][item_id];
            }
            else {
                int item = rand_items[user];
                small_item_id[user] = item;
                std::vector<int>::iterator it = std::find(_user_likes[user].begin(), _user_likes[user].end(), item);
                if (it == _user_likes[user].end()) {
                    small_preference[user] = 0;
                }
                else {
                    int item_id = std::distance(_user_likes[user].begin(), it);
                    small_preference[user] = 1 + _sgd_alpha * _user_likes_weights[user][item_id];
                }
            }
        }

        double end = get_wall_time();
        transfers += (end - start);
        start = get_wall_time();

#pragma omp parallel for num_threads(omp_get_max_threads())
        for (int id = 0; id < _count_users; id++) {
            update_features(&small_item_id[0], &small_preference[0],
                            &_features_users[0], &_features_items[0], id);
        }

        end = get_wall_time();
        calc += (end - start);
    }
}

void sgd::train_random_preferences_gpu()
{
    size_t cuda_free_mem = 0;
    size_t cuda_total_mem = 0;

    int cur_item_start = 0;

    while (cur_item_start < _count_items) {
        double start = get_wall_time();
        double end;

        int item_left_size = _count_items - cur_item_start;

        cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
        int count_items_current = (int) ((cuda_free_mem * 0.75) / (_count_features * 4));

        count_items_current = count_items_current > item_left_size ? item_left_size : count_items_current;

        thrust::device_vector<float> d_features_items(_features_items.begin() + cur_item_start * _count_features,
                                                      _features_items.begin()
                                                          + (cur_item_start + count_items_current) * _count_features);

        int cur_user_start = 0;

        cudaDeviceSynchronize();
        end = get_wall_time();
        transfers += (end - start);

        std::cout << "--Items size: " << count_items_current << std::endl;

        while (cur_user_start < _count_users) {
            start = get_wall_time();

            int user_left_size = _count_users - cur_user_start;

            cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
            int count_users_current = (int) ((cuda_free_mem * 0.95) / ((_count_features + 5) * 4));
            count_users_current = count_users_current > user_left_size ? user_left_size : count_users_current;

            thrust::device_vector<float> d_features_users(_features_users.begin() + cur_user_start * _count_features,
                                                          _features_users.begin()
                                                              + (cur_user_start + count_users_current)
                                                                  * _count_features);

            dim3 block_2d(BLOCK_SIZE, 64);
            dim3 grid_2d(1 + count_users_current / BLOCK_SIZE, 1 + _count_features / 64);

            cudaDeviceSynchronize();
            end = get_wall_time();
            transfers += (end - start);

            std::cout << "---Users size: " << count_users_current << std::endl;

            if (cudaSuccess != cudaPeekAtLastError())
                std::cout << "!WARN - Cuda thrust error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

            thrust::device_vector<unsigned int> random_data(3 * count_users_current);
            std::vector<unsigned int> random_data_host(3 * count_users_current);

            int batch_iter_count = 10;
            for (int i = 0; i < batch_iter_count; i++) {
                start = get_wall_time();

                std::vector<int> is_positive_ratings(count_users_current);
                std::vector<int> rand_user_items(count_users_current);
                std::vector<int> rand_items(count_users_current);

                curandGenerate(gen, thrust::raw_pointer_cast(&random_data.front()), 3 * count_users_current);
                thrust::copy(random_data.begin(), random_data.end(), random_data_host.begin());
                cudaDeviceSynchronize();

#pragma omp parallel for num_threads(omp_get_max_threads())
                for (int j = 0; j < count_users_current; j++) {
                    is_positive_ratings[j] = random_data_host[j] % 10;
                    rand_user_items[j] = random_data_host[j + 1] % _user_likes[cur_user_start + j].size();
                    rand_items[j] = random_data_host[j + 2] % count_items_current;
                }

                /*for (int j = 0; j < count_users_current; j++) {
                    is_positive_ratings[j] = fastrand() % 10;
                    rand_user_items[j] = fastrand() % _user_likes[cur_user_start + j].size();
                    rand_items[j] = fastrand() % count_items_current;
                }*/

                std::vector<int> small_item_id(count_users_current);
                std::vector<float> small_preference(count_users_current);

#pragma omp parallel for num_threads(omp_get_max_threads())
                for (int j = 0; j < count_users_current; j++) {
                    int user = cur_user_start + j;
                    if (is_positive_ratings[j] < 1) {
                        int item_id = rand_user_items[j];
                        int item = _user_likes[user][item_id];
                        if (item >= cur_item_start && item < cur_item_start + count_items_current) {
                            small_item_id[j] = item - cur_item_start;
                        }
                        else {
                            small_item_id[j] = -1;
                        }
                        small_preference[j] = 1 + _sgd_alpha * _user_likes_weights[user][item_id];
                    }
                    else {
                        int item = rand_items[j];
                        small_item_id[j] = item;
                        std::vector<int>::iterator
                            it = std::find(_user_likes[user].begin(), _user_likes[user].end(), item + cur_item_start);
                        if (it == _user_likes[user].end()) {
                            small_preference[j] = 0;
                        }
                        else {
                            int item_id = std::distance(_user_likes[user].begin(), it);
                            small_preference[j] = 1 + _sgd_alpha * _user_likes_weights[user][item_id];
                        }
                    }
                }
                thrust::device_vector<int> d_small_item_id(small_item_id);
                thrust::device_vector<float> d_small_preference(small_preference);

                cudaDeviceSynchronize();
                end = get_wall_time();
                transfers += (end - start);

                if (cudaSuccess != cudaPeekAtLastError())
                    std::cout << "!WARN - Cuda thrust error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

                start = get_wall_time();

                update_features_2d_gpu << < grid_2d, block_2d >> > (
                    thrust::raw_pointer_cast(&d_small_item_id[0]), thrust::raw_pointer_cast(&d_small_preference[0]),
                        thrust::raw_pointer_cast(&d_features_users[0]), thrust::raw_pointer_cast(&d_features_items[0]),
                        small_preference.size(), _count_features, _sgd_lambda, _sgd_learning_rate);

                cudaDeviceSynchronize();
                end = get_wall_time();
                calc += (end - start);
                if (cudaSuccess != cudaPeekAtLastError())
                    std::cout << "!WARN - Cuda kernellll error: " << cudaGetErrorString(cudaGetLastError())
                        << std::endl;

            }
            start = get_wall_time();

            thrust::copy(d_features_users.begin(), d_features_users.end(),
                         _features_users.begin() + cur_user_start * _count_features);
            cudaDeviceSynchronize();
            end = get_wall_time();
            transfers += (end - start);

            if (cudaSuccess != cudaPeekAtLastError())
                std::cout << "!WARN - Cuda thrust error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

            cur_user_start += count_users_current;
        }

        start = get_wall_time();

        thrust::copy(d_features_items.begin(), d_features_items.end(),
                     _features_items.begin() + cur_item_start * _count_features);

        cudaDeviceSynchronize();
        end = get_wall_time();
        transfers += (end - start);
        cur_item_start += count_items_current;
    }
}

void sgd::update_features(int *item_ids, float *preferences,
                          float *features_users, float *features_items, int user_id)
{
    float error = preferences[user_id] - get_prediction(user_id, item_ids[user_id], features_users, features_items);

    int user_mul_feat = user_id * _count_features;
    int item_mul_feat = item_ids[user_id] * _count_features;

    for (int i = 0; i < _count_features; i++) {
        float user_feature = features_users[user_mul_feat + i];
        float item_feature = features_items[item_mul_feat + i];

        float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
        float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

        features_users[user_mul_feat + i] = user_feature + (_sgd_learning_rate * delta_user_feature);
        features_items[item_mul_feat + i] = item_feature + (_sgd_learning_rate * delta_item_feature);
    }
}

float sgd::get_prediction(int user, int item, float *features_users, float *features_items)
{
    float ans = 0;
    int user_mul_feat = user * _count_features;
    int item_mul_feat = item * _count_features;
    for (int i = 0; i < _count_features; i++) {
        ans += (features_users[user_mul_feat + i] * features_items[item_mul_feat + i]);
    }
    return ans;
}

float sgd::hit_rate_cpu()
{
    if (!test_set.size()) {
        return 0;
    }
    float tp = 0;
    std::cout << "hr calc started.. ";
    for (int i = 0; i < test_set.size(); i++) {
        int user = test_set[i].first;
        int item = test_set[i].second;
        std::vector<float> predict(_count_items);
#pragma omp parallel for num_threads(omp_get_max_threads())
        for (int j = 0; j < _count_items; j++) {
            float sum = 0;
            for (int k = 0; k < _count_features; k++) {
                sum += _features_users[user * _count_features + k]
                    * _features_items[j * _count_features + k];
            }
            predict[j] = sum;
        }

#pragma omp parallel for num_threads(omp_get_max_threads())
        for (unsigned int j = 0; j < _user_likes[user].size(); j++) {
            int item_id = _user_likes[user][j];
            predict[item_id] = -1000000;
        }

        for (int j = 0; j < 10; j++) {
            std::vector<float>::iterator it = std::max_element(predict.begin(), predict.end());
            int top_item = std::distance(predict.begin(), it);
            predict[top_item] = -1000000;
            if (top_item == item) {
                tp++;
                break;
            }
        }
    }
    std::cout << "done\n";


    float hr10 = tp * 1.0 / test_set.size();

    std::cout << hr10 << std::endl;

    return hr10;
}

void sgd::serialize_users_map(std::ostream &out)
{
    serialize_map(out, _users_map);
}

void sgd::serialize_items_map(std::ostream &out)
{
    serialize_map(out, _items_map);
}

void sgd::serialize_map(std::ostream &out, std::map<unsigned long, int> &out_map)
{
    std::map<unsigned long, int>::iterator it = out_map.begin();
    for (; it != out_map.end(); it++) {
        out << it->first << "\t" << it->second << std::endl;
    }
}

void sgd::serialize_items(std::ostream &out)
{
    const sgd::features_vector &items = get_features_items();
    serialize_matrix(out, &items.front(), _count_features, _count_items, true);
}

void sgd::serialize_users(std::ostream &out)
{
    const sgd::features_vector &users = get_features_users();
    serialize_matrix(out, &users.front(), _count_features, _count_users, true);
}

void sgd::serialize_matrix(std::ostream &out, const float *mat, int crow, int ccol, bool id)
{
    char *buf = (char *) malloc(10 * sizeof(char));
    for (int i = 0; i < ccol; i++) {
        if (id) out << i << "\t";

        for (int j = 0; j < crow; j++) {
            sprintf(buf, "%.1f", mat[i * crow + j]);
            out << buf << ((j == crow - 1) ? "" : "\t");
        }
        out << std::endl;
    }
}