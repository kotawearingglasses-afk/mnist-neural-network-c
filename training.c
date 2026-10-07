#include "mnist_loader.h"
#include <math.h>
#include <stdlib.h>

// 与えられた要素がn個ある配列oの要素それぞれをxにする関数
void init(int n, float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}

// ReLU層(活性化層) 要素がn個ある配列xがReLU層を通過した後の値を配列yに代入
void relu(int n, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = 0;
        }
    }
}

// FC層(y=Ax+b)　要素がn個ある配列xに対してm*n行列Aをかけ合わせ、要素がm個ある配列bを足し合わせる
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    for (int i = 0; i < m; i++)
    {
        float Ax = 0;
        for (int j = 0; j < n; j++)
        {
            Ax += A[i * n + j] * x[j];
        }
        y[i] = Ax + b[i];
    }
}

// ソフトマックス層（出力層）要素がn個ある配列xの各要素のexpをとって正規化したものを配列yに代入する
void softmax(int n, const float *x, float *y)
{
    float sumx = 0, xmax = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x[i] > xmax)
        {
            xmax = x[i];
        }
    }
    for (int i = 0; i < n; i++)
    {
        sumx += expf(x[i] - xmax);
    }
    for (int i = 0; i < n; i++)
    {
        y[i] = expf(x[i] - xmax) / sumx;
    }
}

// NNでの推論
// 要素が784個の配列xの入力に対してA、bを用いたFC層、RuLU層、A2,b2を用いたFC層、ReLU層、A3,b3を用いたFC層、ソフトマックス層を順に通過させる
// 戻り値に最も確率の高い[0~9]の数字を返す
int inference6(const float *A, const float *b, const float *x, const float *A2, const float *b2, const float *A3, const float *b3, float *y)
{
    float *fc1_y = malloc(sizeof(float) * 50);
    float *relu1_y = malloc(sizeof(float) * 50);
    float *fc2_y = malloc(sizeof(float) * 100);
    float *relu2_y = malloc(sizeof(float) * 100);
    float *fc3_y = malloc(sizeof(float) * 10);
    fc(50, 784, x, A, b, fc1_y);
    relu(50, fc1_y, relu1_y);
    fc(100, 50, relu1_y, A2, b2, fc2_y);
    relu(100, fc2_y, relu2_y);
    fc(10, 100, relu2_y, A3, b3, fc3_y);
    softmax(10, fc3_y, y);
    int max_p = 0;
    for (int i = 1; i < 10; i++)
    {
        if (y[max_p] < y[i])
        {
            max_p = i;
        }
    }
    free(fc1_y);
    free(fc2_y);
    free(fc3_y);
    free(relu1_y);
    free(relu2_y);
    return max_p;
}

// 損失関数＋ソフトマックス層の誤差逆伝搬
// 要素がn個ある配列y、と答えの整数tから勾配dE/dxを求め代入
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    for (int i = 0; i < n; i++)
    {
        if (i == t)
        {
            dEdx[i] = y[i] - 1;
        }
        else
        {
            dEdx[i] = y[i];
        }
    }
}

// relu層の誤差逆伝搬
// 要素がn個ある配列x、dEdyから勾配dE/dxを求め代入する
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)
{
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
        {
            dEdx[i] = dEdy[i];
        }
        else
        {
            dEdx[i] = 0;
        }
    }
}

// fc層の誤差逆伝搬
// 要素がm個の配列dEdy、要素がn個の配列xから勾配dE/dAを求め代入
// 要素がm個の配列dEdyから勾配dE/dbを求め代入
// 要素n*m個の配列A、要素m個の配列dEdyから勾配dE/dxを求め代入
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
    }
    for (int i = 0; i < m; i++)
    {
        dEdb[i] = dEdy[i];
    }
    for (int i = 0; i < n; i++)
    {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            dEdx[i] += A[j * n + i] * dEdy[j];
        }
    }
}

// 与えられた要素がn個ある配列xの要素をランダムにシャッフルする関数
void shuffle(int n, int *x)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int t = x[i];
        x[i] = x[j];
        x[j] = t;
    }
}

// 与えられた要素がn個ある配列o,xに対してそれぞれの要素を加算し、加算した値をoの要素にする関数
void add(int n, const float *x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = o[i] + x[i];
    }
}

// 与えられた要素がn個ある配列oの要素それぞれをx倍する関数
void scale(int n, float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = o[i] * x;
    }
}

// 与えられた要素がn個ある配列oの要素それぞれを範囲[-1,1]の値にランダムに初期化する関数
void rand_init(int n, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }
}

// He初期化 要素がn個あるパラメータAに対して、Aの要素が平均0、分散(2/pre_nord)^(1/2)になるように初期化する関数（pre_nordはパラメータを使う前の層の出力数）
void rand_init_He(int n, int pre_nord, float *A)
{
    float variance = sqrt(2.0 / pre_nord);

    // ボックスミューラー法
    for (int i = 0; i < n; i++)
    {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        A[i] = sqrt(-2.0f * log(u1)) * cosf(2.0f * 3.14159265358979323846264338327950288 * u2) * variance;
    }
}

// NNの出力yと正解tから損失関数を求める関数
float cross_entropy_error(const float *y, int t)
{
    return (-1) * log(y[t] + 1e-7);
}

// NNの誤差逆伝播を行い、平均勾配を求める関数
void backward6(const float *A, const float *b, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)
{
    float *fc1_y = malloc(sizeof(float) * 50);
    float *relu1_y = malloc(sizeof(float) * 50);
    float *fc2_y = malloc(sizeof(float) * 100);
    float *relu2_y = malloc(sizeof(float) * 100);
    float *fc3_y = malloc(sizeof(float) * 10);
    float *softmax_dEdx = malloc(sizeof(float) * 10);
    float *fc3_dEdx = malloc(sizeof(float) * 100);
    float *relu2_dEdx = malloc(sizeof(float) * 100);
    float *fc2_dEdx = malloc(sizeof(float) * 50);
    float *relu1_dEdx = malloc(sizeof(float) * 50);
    float *fc1_dEdx = malloc(sizeof(float) * 784);

    fc(50, 784, x, A, b, fc1_y);
    relu(50, fc1_y, relu1_y);
    fc(100, 50, relu1_y, A2, b2, fc2_y);
    relu(100, fc2_y, relu2_y);
    fc(10, 100, relu2_y, A3, b3, fc3_y);
    softmax(10, fc3_y, y);

    init(10, 0, softmax_dEdx);

    softmaxwithloss_bwd(10, y, t, softmax_dEdx);
    fc_bwd(10, 100, relu2_y, softmax_dEdx, A3, dEdA3, dEdb3, fc3_dEdx);
    relu_bwd(100, relu2_y, fc3_dEdx, relu2_dEdx);
    fc_bwd(100, 50, relu1_y, relu2_dEdx, A2, dEdA2, dEdb2, fc2_dEdx);
    relu_bwd(50, relu1_y, fc2_dEdx, relu1_dEdx);
    fc_bwd(50, 784, x, relu1_dEdx, A, dEdA, dEdb, fc1_dEdx);
    free(fc1_y);
    free(fc2_y);
    free(fc3_y);
    free(relu1_y);
    free(relu2_y);
    free(softmax_dEdx);
    free(relu1_dEdx);
    free(relu2_dEdx);
    free(fc1_dEdx);
    free(fc2_dEdx);
    free(fc3_dEdx);
}

// 各fc層のパラメータを保存する関数
// m*n個要素を持つ配列Aとm個要素を持つ配列bをfilenameを名前にもつファイルで保存
void save(const char *filename, int m, int n,
          const float *A, const float *b)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("file cannot open!\n");
    }
    else
    {
        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), m, fp);
        fclose(fp);
    }
}

// アダマール積　要素がそれぞれn個ある配列o、xについてそれぞれの要素をかけ合わせる
void hadamard_product(int n, const float *o, float *x)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = x[i] * o[i];
    }
}

// 配列コピー　要素がそれぞれn個ある配列o、xについて配列xに配列oをコピーする
void copy_array(int n, const float *o, float *x)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = o[i];
    }
}

// 平均勾配ave_gradからモーメントfirst_moment、second_momentを更新し、モーメントを実行エポック数epoch_numberごとにバイアスをかけ、モーメントを用いてm×nの大きさを持つパラメータparaを更新する
void adam(int m, int n, float *para, float *ave_grad, float *first_moment, float *second_moment, const int epoch_number)
{
    // ハイパーパラメータの設定
    float learning_rate = 0.005f;
    float first_decline_rate = 0.9f;
    float second_decline_rate = 0.999f;
    float epsilon = 1e-8f;

    // バイアス補正後のモーメント
    float *biased_first_moment = malloc(sizeof(float) * m * n);
    float *biased_second_moment = malloc(sizeof(float) * m * n);

    // モーメントの更新
    float *ave_grad_copy = malloc(sizeof(float) * m * n);
    copy_array(m * n, ave_grad, ave_grad_copy);
    scale(m * n, first_decline_rate, first_moment);
    scale(m * n, (1.0f - first_decline_rate), ave_grad_copy);
    add(m * n, ave_grad_copy, first_moment);
    copy_array(m * n, ave_grad, ave_grad_copy);
    scale(m * n, second_decline_rate, second_moment);
    hadamard_product(m * n, ave_grad_copy, ave_grad_copy);
    scale(m * n, (1.0f - second_decline_rate), ave_grad_copy);
    add(m * n, ave_grad_copy, second_moment);

    // バイアス補正
    copy_array(m * n, first_moment, biased_first_moment);
    copy_array(m * n, second_moment, biased_second_moment);
    scale(m * n, 1.0f / (1.0f - powf(first_decline_rate, epoch_number)), biased_first_moment);
    scale(m * n, 1.0f / (1.0f - powf(second_decline_rate, epoch_number)), biased_second_moment);

    // パラメータ更新
    for (int i = 0; i < m * n; i++)
    {
        if (biased_second_moment[i] < 0)
        {
            printf("error");
            break;
        }
        else
        {
            float update = learning_rate * biased_first_moment[i] / (sqrtf(biased_second_moment[i]) + epsilon);
            para[i] -= update;
        }
    }

    free(biased_first_moment);
    free(biased_second_moment);
    free(ave_grad_copy);
}

int main()
{
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);
    int epoch = 10;                        // エポック数
    int mini_batch_size = 100;             // ミニバッヂ数
    float *y = malloc(sizeof(float) * 10); // NNの出力するで0~9である確率
    // 各fc層のパラメータ
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    int *train_index = malloc(sizeof(int) * train_count); // データシャッフル用の配列
    // 各fc層の勾配とその平均
    float *dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *dEdb1 = malloc(sizeof(float) * 50);
    float *ave_dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *ave_dEdb1 = malloc(sizeof(float) * 50);
    float *dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *dEdb2 = malloc(sizeof(float) * 100);
    float *ave_dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *ave_dEdb2 = malloc(sizeof(float) * 100);
    float *dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *dEdb3 = malloc(sizeof(float) * 10);
    float *ave_dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *ave_dEdb3 = malloc(sizeof(float) * 10);
    float cross_entropy; // 各エポックの損失関数の和
    int success;         // 各エポックのNNの正解数
    int epochnumber;     // エポックの試行回数
    // Adam用のパラメータ
    float *first_moment_A1 = calloc(784 * 50, sizeof(float));
    float *first_moment_b1 = calloc(50, sizeof(float));
    float *second_moment_A1 = calloc(784 * 50, sizeof(float));
    float *second_moment_b1 = calloc(50, sizeof(float));

    float *first_moment_A2 = calloc(50 * 100, sizeof(float));
    float *first_moment_b2 = calloc(100, sizeof(float));
    float *second_moment_A2 = calloc(50 * 100, sizeof(float));
    float *second_moment_b2 = calloc(100, sizeof(float));

    float *first_moment_A3 = calloc(100 * 10, sizeof(float));
    float *first_moment_b3 = calloc(10, sizeof(float));
    float *second_moment_A3 = calloc(100 * 10, sizeof(float));
    float *second_moment_b3 = calloc(10, sizeof(float));

    // A,b,A2,b2,A3,b3 の値の初期化
    rand_init_He(784 * 50, 784, A1);
    rand_init(50, b1);
    rand_init_He(50 * 100, 50, A2);
    rand_init(100, b2);
    rand_init_He(100 * 10, 100, A3);
    rand_init(10, b3);

    // 各エポックの試行（訓練）
    epochnumber = 1; // エポック数の初期化
    while (epoch > 0)
    {
        // 学習データでの学習

        // indexの要素の初期化
        for (int i = 0; i < train_count; i++)
        {
            train_index[i] = i;
        }
        // indexの要素をランダムに入れ替える
        shuffle(train_count, train_index);
        // 成功数、損失関数の和の初期化
        success = 0;
        cross_entropy = 0;

        // ミニバッヂごとの試行
        for (int i = 0; i < train_count; i += mini_batch_size)
        {
            // 各FC層のパラメータ更新用の平均勾配dEdA,dEdbを初期化
            init(784 * 50, 0, ave_dEdA1);
            init(50, 0, ave_dEdb1);
            init(50 * 100, 0, ave_dEdA2);
            init(100, 0, ave_dEdb2);
            init(100 * 10, 0, ave_dEdA3);
            init(10, 0, ave_dEdb3);

            // ミニバッヂの各要素に対しての試行
            for (int j = i; j < i + mini_batch_size; j++)
            {
                // NNで要素に対して推論を行い、正解ならば正解数を増やす
                if (inference6(A1, b1, train_x + 784 * train_index[j], A2, b2, A3, b3, y) == train_y[train_index[j]])
                {
                    success++;
                }

                // 誤差逆伝搬を行い各FC層のdEdA,dEdbを更新
                backward6(A1, b1, A2, b2, A3,
                          b3, train_x + 784 * train_index[j], train_y[train_index[j]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);

                // 各dEdA、dEdbをパラメータ更新用の変数に加算する(後に更新用変数は全要素数で割られ平均となる)
                add(784 * 50, dEdA1, ave_dEdA1);
                add(50, dEdb1, ave_dEdb1);
                add(50 * 100, dEdA2, ave_dEdA2);
                add(100, dEdb2, ave_dEdb2);
                add(100 * 10, dEdA3, ave_dEdA3);
                add(10, dEdb3, ave_dEdb3);

                // 損失関数の和に要素の損失関数を加算する
                cross_entropy += cross_entropy_error(y, train_y[train_index[j]]);
            }

            // 平均勾配を計算し、それらをもとにAdamでパラメータを更新
            scale(784 * 50, 1.0 / mini_batch_size, ave_dEdA1);
            scale(50, 1.0 / mini_batch_size, ave_dEdb1);
            adam(784, 50, A1, ave_dEdA1, first_moment_A1, second_moment_A1, epochnumber);
            adam(50, 1, b1, ave_dEdb1, first_moment_b1, second_moment_b1, epochnumber);
            scale(50 * 100, 1.0 / mini_batch_size, ave_dEdA2);
            scale(100, 1.0 / mini_batch_size, ave_dEdb2);
            adam(50, 100, A2, ave_dEdA2, first_moment_A2, second_moment_A2, epochnumber);
            adam(100, 1, b2, ave_dEdb2, first_moment_b2, second_moment_b2, epochnumber);
            scale(100 * 10, 1.0 / mini_batch_size, ave_dEdA3);
            scale(10, 1.0 / mini_batch_size, ave_dEdb3);
            adam(100, 10, A3, ave_dEdA3, first_moment_A3, second_moment_A3, epochnumber);
            adam(10, 1, b3, ave_dEdb3, first_moment_b3, second_moment_b3, epochnumber);
        }

        // エポック数と損失関数の平均、正解率を表示
        printf("train: epoch number %d: loss %f ,accuracy %f%% \n", epochnumber, cross_entropy / train_count, success * 100.0 / train_count);

        // テストデータに対して推論

        // 成功数、損失関数の和の初期化
        success = 0;
        cross_entropy = 0;

        // 各テストデータをNNにかけ、正答率と損失関数を出す
        for (int i = 0; i < test_count; i++)
        {
            if (inference6(A1, b1, test_x + 784 * i, A2, b2, A3, b3, y) == test_y[i])
            {
                success++;
            }
            cross_entropy += cross_entropy_error(y, test_y[i]);
        }

        // エポック数と損失関数の平均、正解率を表示
        printf("test: epoch number %d: loss %f ,accuracy %f%% \n", epochnumber, cross_entropy / test_count, success * 100.0 / test_count);

        // エポック数を加算、残エポック数を減算
        epochnumber++;
        epoch--;
    }
    // 各A,bを保存する
    save("fc1_parameter_adam.dat", 50, 784, A1, b1);
    save("fc2_parameter_adam.dat", 100, 50, A2, b2);
    save("fc3_parameter_adam.dat", 10, 100, A3, b3);

    // 確保したメモリの解放
    free(A1);
    free(b1);
    free(first_moment_A1);
    free(second_moment_A1);
    free(first_moment_b1);
    free(second_moment_b1);
    free(A2);
    free(b2);
    free(first_moment_A2);
    free(second_moment_A2);
    free(first_moment_b2);
    free(second_moment_b2);
    free(A3);
    free(b3);
    free(first_moment_A3);
    free(second_moment_A3);
    free(first_moment_b3);
    free(second_moment_b3);
    free(dEdA1);
    free(dEdb1);
    free(dEdA2);
    free(dEdb2);
    free(dEdA3);
    free(dEdb3);
    free(ave_dEdA1);
    free(ave_dEdb1);
    free(ave_dEdA2);
    free(ave_dEdb2);
    free(ave_dEdA3);
    free(ave_dEdb3);
    free(y);
    return 0;
}