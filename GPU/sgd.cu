#include "sgd.cuh"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#define BLOCK_SIZE 8

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


//    srand(time(NULL));
    srand(34);


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

void sgd::calculate(int count_iterations, int positive_ratings, int negative_ratings)
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


    for (int i = 0; i < count_iterations; i++) {
        double start = get_wall_time();
        std::cerr << "SGD Iteration: " << i << std::endl;

        train_random_preferences();



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


        double end = get_wall_time();
        std::cerr << "==== Iteration time : " << (end - start) / 1000000 << std::endl;


        /*old_hr = new_hr;
        new_hr = hit_rate_cpu();
        if (new_hr > old_hr) {
            _sgd_learning_rate *= 1.05;
        }
        else {
            _sgd_learning_rate *= 0.5;
        }*/
        hr10 << new_hr << std::endl;

    }

    std::cout << "Profiler:\n";
    std::cout << "Transfers: " << transfers / 1000000 << std::endl;
    std::cout << "Calc: " << calc / 1000000 << std::endl;


    hr10.close();

}

__global__ void update_features_gpu(int *user_ids, int *item_ids, float *preferences,
                                    float *features_users, float *features_items, int ratings_count,
                                    int _count_features, float _sgd_lambda, float _sgd_learning_rate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ratings_count) {
        int user_mul_feat = user_ids[idx] * _count_features;
        int item_mul_feat = item_ids[idx] * _count_features;

        float prediction = 0;
        for (int i = 0; i < _count_features; i++) {
            prediction += (features_users[user_mul_feat + i] * features_items[item_mul_feat + i]);
        }

        float error = preferences[idx] - prediction;

        for (int i = 0; i < _count_features; i++) {
            float user_feature = features_users[user_mul_feat + i];
            float item_feature = features_items[item_mul_feat + i];

            float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
            float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

            features_users[user_mul_feat + i] = user_feature + (_sgd_learning_rate * delta_user_feature);
            features_items[item_mul_feat + i] = item_feature + (_sgd_learning_rate * delta_item_feature);
        }
    }
}

void sgd::train_random_preferences()
{
    double start = get_wall_time();

    std::vector<int> small_user_id;
    std::vector<int> small_item_id;
    std::vector<float> small_preference;
    features_vector small_features_users;
    features_vector small_features_items;

    std::vector<int> user_id_to_small(_count_users, -1);
    std::vector<int> user_id_from_small;
    std::vector<int> item_id_to_small(_count_items, -1);
    std::vector<int> item_id_from_small;

    int count_small_users = 0;
    int count_small_items = 0;


    for (int i = 0; i < _positive_ratings; i++) {
        int user = rand() % _user_likes.size();
        int item_id = rand() % _user_likes[user].size();
        int item = _user_likes[user][item_id];
        float preference = 1 + _sgd_alpha * _user_likes_weights[user][item_id];

        if (user_id_to_small[user] == -1) {
            user_id_to_small[user] = count_small_users;
            user_id_from_small.push_back(user);
            /*small_features_users.insert(small_features_users.end(),
                                        _features_users.begin() + user * _count_features,
                                        _features_users.begin() + (user + 1) * _count_features);*/
            count_small_users++;
        }
        if (item_id_to_small[item] == -1) {
            item_id_to_small[item] = count_small_items;
            item_id_from_small.push_back(item);
            /*small_features_items.insert(small_features_items.end(),
                                        _features_items.begin() + item * _count_features,
                                        _features_items.begin() + (item + 1) * _count_features);*/
            count_small_items++;
        }

        small_user_id.push_back(user_id_to_small[user]);
        small_item_id.push_back(item_id_to_small[item]);
        small_preference.push_back(preference);
    }

    for (int i = 0; i < _negative_ratings; i++) {
        int user = rand() % _user_likes.size();
        int item = rand() % _item_likes.size();
        if (std::find(_user_likes[user].begin(), _user_likes[user].end(), item) == _user_likes[user].end()) {
            float preference = 0;

            if (user_id_to_small[user] == -1) {
                user_id_to_small[user] = count_small_users;
                user_id_from_small.push_back(user);
                /*small_features_users.insert(small_features_users.end(),
                                            _features_users.begin() + user * _count_features,
                                            _features_users.begin() + (user + 1) * _count_features);*/
                count_small_users++;
            }
            if (item_id_to_small[item] == -1) {
                item_id_to_small[item] = count_small_items;
                item_id_from_small.push_back(item);
                /*small_features_items.insert(small_features_items.end(),
                                            _features_items.begin() + item * _count_features,
                                            _features_items.begin() + (item + 1) * _count_features);*/
                count_small_items++;
            }

            small_user_id.push_back(user_id_to_small[user]);
            small_item_id.push_back(item_id_to_small[item]);
            small_preference.push_back(preference);
        }
    }

    small_features_users.resize(count_small_users * _count_features);
    small_features_items.resize(count_small_items * _count_features);

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < count_small_users; i++) {
        int user_id = user_id_from_small[i];
        std::copy(_features_users.begin() + user_id * _count_features,
                  _features_users.begin() + (user_id + 1) * _count_features,
                  small_features_users.begin() + i * _count_features);
    }

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < count_small_items; i++) {
        int item_id = item_id_from_small[i];
        std::copy(_features_items.begin() + item_id * _count_features,
                  _features_items.begin() + (item_id + 1) * _count_features,
                  small_features_items.begin() + i * _count_features);
    }
    /*double end = get_wall_time();
    transfers += (end - start);
    start = get_wall_time();*/



    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(1 + small_preference.size() / BLOCK_SIZE, 1);

    thrust::device_vector<int> d_small_user_id(small_user_id);
    thrust::device_vector<int> d_small_item_id(small_item_id);
    thrust::device_vector<float> d_small_preference(small_preference);
    thrust::device_vector<float> d_small_features_users(small_features_users);
    thrust::device_vector<float> d_small_features_items(small_features_items);

    cudaDeviceSynchronize();

    double end = get_wall_time();
    transfers += (end - start);
    start = get_wall_time();



    update_features_gpu << < grid, block >> > (thrust::raw_pointer_cast(&d_small_user_id[0]),
        thrust::raw_pointer_cast(&d_small_item_id[0]), thrust::raw_pointer_cast(&d_small_preference[0]),
        thrust::raw_pointer_cast(&d_small_features_users[0]), thrust::raw_pointer_cast(&d_small_features_items[0]),
        small_preference.size(), _count_features, _sgd_lambda, _sgd_learning_rate);

    cudaDeviceSynchronize();
    end = get_wall_time();
    calc += (end - start);
    start = get_wall_time();


/*
#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < small_preference.size(); i++) {
        update_features_small(&small_user_id[0], &small_item_id[0], &small_preference[0],
                              &small_features_users[0], &small_features_items[0], i);
    }

    end = get_wall_time();
    calc += (end - start);
    start = get_wall_time();

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < user_id_from_small.size(); i++) {
        std::copy(small_features_users.begin() + i * _count_features,
                  small_features_users.begin() + (i + 1) * _count_features,
                  _features_users.begin() + user_id_from_small[i] * _count_features);
    }

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < item_id_from_small.size(); i++) {
        std::copy(small_features_items.begin() + i * _count_features,
                  small_features_items.begin() + (i + 1) * _count_features,
                  _features_items.begin() + item_id_from_small[i] * _count_features);
    }
    end = get_wall_time();
    transfers += (end - start);
*/


    thrust::host_vector<float> h_small_features_users(d_small_features_users);
    thrust::host_vector<float> h_small_features_items(d_small_features_items);

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < user_id_from_small.size(); i++) {
        thrust::copy(h_small_features_users.begin() + i * _count_features,
                     h_small_features_users.begin() + (i + 1) * _count_features,
                     _features_users.begin() + user_id_from_small[i] * _count_features);
    }

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < item_id_from_small.size(); i++) {
        thrust::copy(h_small_features_items.begin() + i * _count_features,
                     h_small_features_items.begin() + (i + 1) * _count_features,
                     _features_items.begin() + item_id_from_small[i] * _count_features);
    }

    cudaDeviceSynchronize();
    end = get_wall_time();
    transfers += (end - start);



}

void sgd::update_features_small(int *user_ids, int *item_ids, float *preferences,
                                float *features_users, float *features_items, int idx)
{
    float error = preferences[idx] - get_prediction_small(user_ids[idx], item_ids[idx], features_users, features_items);

    for (int i = 0; i < _count_features; i++) {
        float user_feature = features_users[user_ids[idx] * _count_features + i];
        float item_feature = features_items[item_ids[idx] * _count_features + i];

        float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
        float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

        features_users[user_ids[idx] * _count_features + i] += (_sgd_learning_rate * delta_user_feature);
        features_items[item_ids[idx] * _count_features + i] += (_sgd_learning_rate * delta_item_feature);
    }
}

float sgd::get_prediction_small(int user, int item, float *features_users, float *features_items)
{
    float ans = 0;
    for (int i = 0; i < _count_features; i++) {
        ans += (features_users[user * _count_features + i] * features_items[item * _count_features + i]);
    }
    return ans;
}

void sgd::update_features(int user, int item, float preference)
{
    float error = preference - get_prediction(user, item);


    for (int i = 0; i < _count_features; i++) {
        float user_feature = _features_users[user * _count_features + i];
        float item_feature = _features_items[item * _count_features + i];

        float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
        float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

        _features_users[user * _count_features + i] += (_sgd_learning_rate * delta_user_feature);
        _features_items[item * _count_features + i] += (_sgd_learning_rate * delta_item_feature);
    }
}

float sgd::get_prediction(int user, int item)
{
    float ans = 0;
    for (int i = 0; i < _count_features; i++) {
        ans += (_features_users[user * _count_features + i] * _features_items[item * _count_features + i]);
    }
    return ans;
}

float sgd::hit_rate_cpu()
{
    if (!test_set.size()) {
        return 0;
    }
    float tp = 0;
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