#ifndef CPU_SGD_H
#define CPU_SGD_H

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <omp.h>


class sgd
{
public:
    ///
    /// Definition of features vector
    ///
    typedef std::vector<float> features_vector;
    typedef std::vector<std::vector<int> > likes_vector;
    typedef std::vector<std::vector<float> > likes_weights_vector;
    typedef std::vector<int> likes_vector_item;
    typedef std::vector<float> likes_weights_vector_item;
    ///
    /// Ctor
    /// Inputs are:
    /// stream with triplets:
    /// count_features - count latent features
    /// format of likes
    /// 0 - old
    /// 1 - simple
    /// <user> <item> <weight>
    ///
    sgd(std::istream &tuples_stream,
        int count_features,
        float learning_rate,
        float lambda,
        float alpha,
        int count_samples,
        int likes_format);


    virtual ~sgd();

    ///
    /// Calculate als (Matrix Factorization)
    /// in
    /// count_iterations - count iterations
    ///
    virtual void calculate(int count_iterations, int positive_ratings, int negative_ratings, int is_gpu_calc);

    virtual float hit_rate_cpu();

    ///
    /// Get Items features vector
    ///
    const features_vector &get_features_items() const
    { return _features_items; }
    int get_count_items() const
    { return _count_items; }

    ///
    /// Get Users features vector
    ///
    const features_vector &get_features_users() const
    { return _features_users; }
    int get_count_users() const
    { return _count_users; }

    void serialize_map(std::ostream &out, std::map<unsigned long, int> &out_map);
    void serialize_matrix(std::ostream &out, const float *mat, int crow, int ccol, bool id = false);
    void serialize_users(std::ostream &out);
    void serialize_items(std::ostream &out);
    void serialize_users_map(std::ostream &out);
    void serialize_items_map(std::ostream &out);

protected:
    ///
    /// Read likes from stream
    /// if format == 0
    /// user group item
    /// if format == 1
    /// user item weight
    ///
    void read_likes(std::istream &tuples_stream, int count_simples, int format);

    ///
    /// fill random values to features matrix
    ///
    void fill_rnd(features_vector &in_v, int in_size);

    void train_random_preferences_cpu();

    void train_random_preferences_gpu();

    void update_features_avg(int user, int item, float preference);

    void update_features(int *item_ids, float *preferences,
                         float *features_users, float *features_items, int idx);

    float get_prediction(int user, int item, float *features_users, float *features_items);

    void generate_test_set();

    void getHRFromDato();

private:
    ///
    /// features vectors, for users and items
    ///
    features_vector _features_users;
    int _count_users;
    features_vector _features_items;
    int _count_items;

    int _count_features;

    ///
    /// Internal data
    ///
    std::map<unsigned long, int> _users_map;
    std::map<unsigned long, int> _items_map;
    likes_vector _user_likes;
    likes_weights_vector _user_likes_weights;
    likes_vector _item_likes;
    likes_weights_vector _item_likes_weights;

    float _sgd_learning_rate;
    float _sgd_lambda;
    float _sgd_alpha;

    int _positive_ratings;
    int _negative_ratings;


    std::vector<std::pair<int, int> > test_set;
    likes_weights_vector _user_likes_weights_temp;

    std::vector<std::pair<int, int> > prefs;

    double transfers;
    double calc;


};

double get_wall_time();

#endif //CPU_SGD_H
