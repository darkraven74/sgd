#include "sgd.h"
#include <cmath>
#include <algorithm>
#include <ctime>
#include <set>

sgd::sgd(std::istream &tuples_stream,
         int count_features,
         float learning_rate,
         float lambda,
         int count_samples,
         int likes_format)
    :
    _count_users(0),
    _count_items(0),
    _count_features(count_features),
    _sgd_learning_rate(learning_rate),
    _sgd_lambda(lambda)
{


//    srand(time(NULL));
    srand(34);


    read_likes(tuples_stream, count_samples, likes_format);

    generate_test_set();

    //getHRFromDato();

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
    //for (int idx = 0; idx < 10000; idx++)
    for (int i = 0; i < _count_users; i++) {
        //int i = rand() % _count_users;
        total_size += _user_likes[i].size();
        int size = _user_likes[i].size();
        for (int j = 0; j < size / 2;) {
            int id = rand() % _user_likes[i].size();

            if (_user_likes_weights_temp[i][id] < 4) {
                continue;
            }
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

    std::ofstream test_ml100k("test_ml100k.txt");
    for (int i = 0; i < test_set.size(); i++) {
        int user = test_set[i].first;
        int item = test_set[i].second;
        test_ml100k << user << "," << item << ",6" << std::endl;
    }
    test_ml100k.close();

    std::ofstream train_ml100k("train_ml100k.txt");
    for (int i = 0; i < _user_likes.size(); i++) {
        for (int j = 0; j < _user_likes[i].size(); j++) {
            train_ml100k << i << "," << _user_likes[i][j] << ",6" << std::endl;
            prefs.push_back(std::make_pair(i, _user_likes[i][j]));
        }
    }

    /*srand(34);
    int i = 0;
    while (i < 1000000) {
        int user = rand() % _user_likes.size();
        int item = rand() % _item_likes.size();
        if (std::find(_user_likes[user].begin(), _user_likes[user].end(), item) != _user_likes[user].end()) {
            continue;
        }
        train_ml100k << user << "," << item << ",0" << std::endl;
        i++;
    }*/
    train_ml100k.close();

}

void sgd::getHRFromDato()
{
    std::ifstream recs_ml100k("recs_dato.csv");
    std::string line;
    getline(recs_ml100k, line);
    char csv_delim = ',';
    std::set<std::pair<int, int> > recs;
    while (getline(recs_ml100k, line)) {
        std::istringstream line_stream(line);
        std::string value;
        getline(line_stream, value, csv_delim);
        unsigned long uid = atol(value.c_str());
        getline(line_stream, value, csv_delim);
        unsigned long iid = atol(value.c_str());
        recs.insert(std::make_pair(uid, iid));
    }

    float tp = 0;
    for (int i = 0; i < test_set.size(); i++) {
        if (recs.count(test_set[i])) {
            tp++;
        }
    }

    std::cout << "*************************** HR10: " << tp / test_set.size() << std::endl;



    recs_ml100k.close();
}

void sgd::fill_rnd(features_vector &in_v, int in_size)
{
    std::cerr << "Generate random features.. ";
    for (int i = 0; i < in_size * _count_features; i++) {
        in_v[i] = ((float) rand() / (float) RAND_MAX);
    }

    std::cerr << "done" << std::endl;
}

void sgd::calculate(int count_iterations)
{
    fill_rnd(_features_users, _count_users);
    fill_rnd(_features_items, _count_items);

    std::ofstream hr10("hr10.txt");

    for (int i = 0; i < count_iterations; i++) {
        time_t start = time(0);
        std::cerr << "SGD Iteration: " << i << std::endl;

//        train_all_preferences();
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


        time_t end = time(0);
        std::cerr << "==== Iteration time : " << end - start << std::endl;

        hr10 << hit_rate_cpu() << std::endl;
    }

    hr10.close();

}

void sgd::train_all_preferences() {
    for (int user = 0; user < _user_likes.size(); user++) {
        for (int i = 0; i < _user_likes[user].size(); i++) {
            int item = _user_likes[user][i];
            float preference = _user_likes_weights[user][i];
            update_features(user, item, 6);
        }
    }

    srand(34);
    int i = 0;
    while (i < 1000000) {
        int user = rand() % _user_likes.size();
        int item = rand() % _item_likes.size();
        if (std::find(_user_likes[user].begin(), _user_likes[user].end(), item) != _user_likes[user].end()) {
            continue;
        }
        update_features(user, item, 0);
        i++;
    }
}

void sgd::train_random_preferences() {
    _features_users_diff.assign(_count_users * _count_features, 0);
    _features_items_diff.assign(_count_items * _count_features, 0);
    _features_users_diff_count.assign(_count_users * _count_features, 0);
    _features_items_diff_count.assign(_count_items * _count_features, 0);
//#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < 99980; i++) {
        int user = rand() % _user_likes.size();
        int item_id = rand() % _user_likes[user].size();
        int item = _user_likes[user][item_id];
        float preference = _user_likes_weights[user][item_id];
        update_features(user, item, 6);
//        update_features_avg(user, item, 6);
    }

    /*srand(time(NULL));
    std::random_shuffle(prefs.begin(), prefs.end());
    for (int i = 0; i < 99980; i++) {
        update_features(prefs[i].first, prefs[i].second, 6);
    }*/

    //srand(34);
//#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < 1000000; i++) {
        int user = rand() % _user_likes.size();
        int item = rand() % _item_likes.size();
        if (std::find(_user_likes[user].begin(), _user_likes[user].end(), item) == _user_likes[user].end()) {
            update_features(user, item, 0);
//            update_features_avg(user, item, 0);
        }
    }

    /*for (int i = 0; i < _count_users; i++) {
        if (_features_users_diff_count[i * _count_features] != 0) {
            for (int k = 0; k < _count_features; k++) {
                int idx = i * _count_features + k;
                _features_users[idx] +=
                    ((_features_users_diff[idx]) / (_features_users_diff_count[idx]));
            }
        }
    }
    for (int i = 0; i < _count_items; i++) {
        if (_features_items_diff_count[i * _count_features] != 0) {
            for (int k = 0; k < _count_features; k++) {
                int idx = i * _count_features + k;
                _features_items[idx] +=
                    ((_features_items_diff[idx]) / (_features_items_diff_count[idx]));
            }
        }
    }*/
}

void sgd::update_features(int user, int item, float preference) {
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

void sgd::update_features_avg(int user, int item, float preference) {
    float error = preference - get_prediction(user, item);


    for (int i = 0; i < _count_features; i++) {
        float user_feature = _features_users[user * _count_features + i];
        float item_feature = _features_items[item * _count_features + i];

        float delta_user_feature = error * item_feature - _sgd_lambda * user_feature;
        float delta_item_feature = error * user_feature - _sgd_lambda * item_feature;

        _features_users_diff[user * _count_features + i] += (_sgd_learning_rate * delta_user_feature);
        _features_items_diff[item * _count_features + i] += (_sgd_learning_rate * delta_item_feature);

        _features_users_diff_count[user * _count_features + i]++;
        _features_items_diff_count[item * _count_features + i]++;

    }
}

float sgd::get_prediction(int user, int item) {
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