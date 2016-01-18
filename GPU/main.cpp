#include <string>

#include <sys/time.h>
#include <iostream>
#include <cstdlib>
#include "sgd.cuh"

int main(int argc, char *argv[])
{
    std::string output_file_name;
    std::string likes_file_name;
    int features_size = 50;
    int csamples = 0;
    int cit = 20;
    int likes_format = 0;
    float learning_rate = 0.01;
//    float learning_rate = 0.1;
    //float lambda = 0.1;
    float lambda = 0.01;
    float alpha = 5;


    int positive_ratings = 100000;
    int negative_ratings = 1000000;

    int is_gpu_calc = 1;


    for (int i = 1; i < argc; i++) {
        std::string sarg = argv[i];
        if (!sarg.compare("--likes")) {
            i++;
            likes_file_name = argv[i];
        }
        else if (!sarg.compare("--f_size")) {
            i++;
            features_size = atoi(argv[i]);
            std::cerr << " Count features:  " << features_size << std::endl;
        }
        else if (!sarg.compare("--csamples")) {
            i++;
            csamples = atoi(argv[i]);
        }
        else if (!sarg.compare("--it")) {
            i++;
            cit = atoi(argv[i]);
        }
        else if (!sarg.compare("--out")) {
            i++;
            output_file_name = argv[i];
        }
        else if (!sarg.compare("--likes-format")) {
            i++;
            likes_format = atoi(argv[i]);
        }
        else if (!sarg.compare("--positive")) {
            i++;
            positive_ratings = atoi(argv[i]);
        }
        else if (!sarg.compare("--negative")) {
            i++;
            negative_ratings = atoi(argv[i]);
        }
        else if (!sarg.compare("--is_gpu_calc")) {
            i++;
            is_gpu_calc = atoi(argv[i]);
        }
    }

    std::ifstream f_stream(likes_file_name.c_str());
    std::istream &in((likes_file_name.length() == 0) ? std::cin : f_stream);

    std::cerr << " Count SGD iteration " << cit << std::endl;
    std::cerr << " Start Matrix Factorization - SGD " << std::endl;
    std::cerr << " Input file format -  " << likes_format << std::endl;


    sgd sgd_alg(in, features_size, learning_rate, lambda, alpha, csamples, likes_format);

    double start = get_wall_time();

    sgd_alg.calculate(cit, positive_ratings, negative_ratings, is_gpu_calc);

    double end = get_wall_time();
    std::cout << "sgd calc time, s: " << (end - start) / 1000000 << std::endl;


    /*std::ofstream fout_users((output_file_name.append(".ufea")).c_str());
    sgd_alg.serialize_users(fout_users);
    fout_users.close();

    std::ofstream fout_items((output_file_name.append(".ifea")).c_str());
    sgd_alg.serialize_items(fout_items);
    fout_items.close();

    std::ofstream fout_umap((output_file_name.append(".umap")).c_str());
    sgd_alg.serialize_users_map(fout_umap);
    fout_umap.close();

    std::ofstream fout_imap((output_file_name.append(".imap")).c_str());
    sgd_alg.serialize_items_map(fout_imap);
    fout_imap.close();
    */


    return 0;
}