#include "mnist_loader.h"
#include <math.h>

// ファイルが開けなかった回数
int file_open_error = 0;

// filenameを名前に持つファイルから大きさm*nの配列をAに、大きさmの配列をbに代入
void load(const char *filename, int m, int n,
          float *A, float *b)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("file cannnot open!\n");
        file_open_error += 1;
    }
    else
    {
        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), m, fp);
        fclose(fp);
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
int inference6(const float *A, const float *b, const float *x, const float *A2, const float *b2, const float *A3, const float *b3)
{
    float *fc1_y = malloc(sizeof(float) * 50);
    float *relu1_y = malloc(sizeof(float) * 50);
    float *fc2_y = malloc(sizeof(float) * 100);
    float *relu2_y = malloc(sizeof(float) * 100);
    float *fc3_y = malloc(sizeof(float) * 10);
    float *softmax_y = malloc(sizeof(float) * 10);
    fc(50, 784, x, A, b, fc1_y);
    relu(50, fc1_y, relu1_y);
    fc(100, 50, relu1_y, A2, b2, fc2_y);
    relu(100, fc2_y, relu2_y);
    fc(10, 100, relu2_y, A3, b3, fc3_y);
    softmax(10, fc3_y, softmax_y);
    int max_p = 0;
    for (int i = 1; i < 10; i++)
    {
        if (softmax_y[max_p] < softmax_y[i])
        {
            max_p = i;
        }
    }
    free(fc1_y);
    free(fc2_y);
    free(fc3_y);
    free(relu1_y);
    free(relu2_y);
    free(softmax_y);
    return max_p;
}

// 実行時引数として推論を行う画像のデータ(.bmp)を与える
int main(int argc, char *argv[])
{
    const char *parameter1 = "fc1_parameter_adam.dat";
    const char *parameter2 = "fc2_parameter_adam.dat";
    const char *parameter3 = "fc3_parameter_adam.dat";

    if (argc != 2) // BMP ファイル名のみを指定する
    {
        printf("Usage: %s <input.bmp>\n", argv[0]);
    }
    else
    {
        // 推論を行うときのFC層の各パラメータ
        float *A1 = malloc(sizeof(float) * 784 * 50);
        float *b1 = malloc(sizeof(float) * 50);
        float *A2 = malloc(sizeof(float) * 50 * 100);
        float *b2 = malloc(sizeof(float) * 100);
        float *A3 = malloc(sizeof(float) * 100 * 10);
        float *b3 = malloc(sizeof(float) * 10);
        // 推論を行う画像（bmp）
        float *x = load_mnist_bmp(argv[1]);
        if (x == NULL)
        {
            return 1;
        }
        // 学習したパラメータをロードし、各A,bに代入
        load(parameter1, 50, 784, A1, b1);
        load(parameter2, 100, 50, A2, b2);
        load(parameter3, 10, 100, A3, b3);

        if (file_open_error == 0) // ファイルのロードを失敗しなかったとき
        {
            // 推論の答えを出力
            printf("inference answer: %d\n", inference6(A1, b1, x, A2, b2, A3, b3));
        }
    }
    return 0;
}
