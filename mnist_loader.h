#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

/*
 * Portfolio version of the MNIST loader.
 *
 * This file replaces the MNIST-related part of the course-provided nn.h.
 * The neural-network implementation itself remains in the user's .c files.
 *
 * Expected MNIST files:
 *   train-images-idx3-ubyte
 *   train-labels-idx1-ubyte
 *   t10k-images-idx3-ubyte
 *   t10k-labels-idx1-ubyte
 *
 * By default they are searched under ./data.
 * Change MNIST_DIR if you want another location.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#ifndef MNIST_DIR
#define MNIST_DIR "./data"
#endif

#ifndef PATH_MAX
#define PATH_MAX 260
#endif

static uint32_t mnist_read_be32(FILE *fp)
{
    unsigned char b[4];

    if (fread(b, 1, 4, fp) != 4) {
        return 0;
    }

    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |
           (uint32_t)b[3];
}

static float *load_mnist_image(const char *filename,
                               int *width, int *height, int *count)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] Cannot read MNIST image file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    uint32_t magic = mnist_read_be32(fp);
    uint32_t ni    = mnist_read_be32(fp);
    uint32_t nr    = mnist_read_be32(fp);
    uint32_t nc    = mnist_read_be32(fp);

    if (magic != 2051 || ni == 0 || nr == 0 || nc == 0) {
        fprintf(stderr, "[ERROR] Invalid MNIST image file: %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    size_t image_size = (size_t)nr * nc;
    size_t total_size = image_size * ni;

    unsigned char *raw = malloc(total_size);
    float *images = malloc(total_size * sizeof(float));

    if (raw == NULL || images == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed while loading %s\n",
                filename);
        free(raw);
        free(images);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (fread(raw, 1, total_size, fp) != total_size) {
        fprintf(stderr, "[ERROR] MNIST image data is incomplete: %s\n", filename);
        free(raw);
        free(images);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    for (size_t i = 0; i < total_size; ++i) {
        images[i] = (float)raw[i] / 255.0f;
    }

    free(raw);

    *width = (int)nc;
    *height = (int)nr;
    *count = (int)ni;

    return images;
}

static unsigned char *load_mnist_label(const char *filename, int *count)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] Cannot read MNIST label file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    uint32_t magic = mnist_read_be32(fp);
    uint32_t ni    = mnist_read_be32(fp);

    if (magic != 2049 || ni == 0) {
        fprintf(stderr, "[ERROR] Invalid MNIST label file: %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    unsigned char *labels = malloc(ni);
    if (labels == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed while loading %s\n",
                filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (fread(labels, 1, ni, fp) != ni) {
        fprintf(stderr, "[ERROR] MNIST label data is incomplete: %s\n", filename);
        free(labels);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    *count = (int)ni;
    return labels;
}

/*
 * Same function signature as the course version, so training_true_adam.c
 * does not need to change its call site.
 */
static void load_mnist(float **train_x, unsigned char **train_y, int *train_count,
                       float **test_x, unsigned char **test_y, int *test_count,
                       int *width, int *height)
{
    char path[PATH_MAX];

    assert(train_x != NULL);
    assert(train_y != NULL);
    assert(train_count != NULL);
    assert(test_x != NULL);
    assert(test_y != NULL);
    assert(test_count != NULL);
    assert(width != NULL);
    assert(height != NULL);

    snprintf(path, sizeof(path), "%s/train-images.idx3-ubyte", MNIST_DIR);
    *train_x = load_mnist_image(path, width, height, train_count);

    if (*width != 28 || *height != 28 || *train_count != 60000) {
        fprintf(stderr,
                "[ERROR] Unexpected training MNIST size: %dx%d, %d images\n",
                *width, *height, *train_count);
        exit(EXIT_FAILURE);
    }

    snprintf(path, sizeof(path), "%s/train-labels.idx1-ubyte", MNIST_DIR);
    *train_y = load_mnist_label(path, train_count);

    if (*train_count != 60000) {
        fprintf(stderr, "[ERROR] Training image/label counts do not match.\n");
        exit(EXIT_FAILURE);
    }

    snprintf(path, sizeof(path), "%s/t10k-images.idx3-ubyte", MNIST_DIR);
    *test_x = load_mnist_image(path, width, height, test_count);

    if (*width != 28 || *height != 28 || *test_count != 10000) {
        fprintf(stderr,
                "[ERROR] Unexpected test MNIST size: %dx%d, %d images\n",
                *width, *height, *test_count);
        exit(EXIT_FAILURE);
    }

    snprintf(path, sizeof(path), "%s/t10k-labels.idx1-ubyte", MNIST_DIR);
    *test_y = load_mnist_label(path, test_count);

    if (*test_count != 10000) {
        fprintf(stderr, "[ERROR] Test image/label counts do not match.\n");
        exit(EXIT_FAILURE);
    }
}

/*
 * Load a simple uncompressed BMP and convert it to a 28x28 grayscale float
 * image in the same 0.0-1.0 format used by MNIST.
 *
 * Supported:
 *   - Windows BMP
 *   - uncompressed 24-bit or 32-bit RGB(A)
 *   - bottom-up and top-down images
 *
 * This replaces the course nn.h dependency on stb_image for the portfolio.
 */
static float *load_mnist_bmp(const char *filename, ...)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] Cannot read BMP file: %s\n", filename);
        return NULL;
    }

    /* BMP file header */
    unsigned char file_header[14];

    if (fread(file_header, 1, 14, fp) != 14 ||
        file_header[0] != 'B' ||
        file_header[1] != 'M') {

        fprintf(stderr, "[ERROR] Not a valid BMP file: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    uint32_t pixel_offset =
        (uint32_t)file_header[10] |
        ((uint32_t)file_header[11] << 8) |
        ((uint32_t)file_header[12] << 16) |
        ((uint32_t)file_header[13] << 24);

    /* DIB header */
    unsigned char dib[40];

    if (fread(dib, 1, 40, fp) != 40) {
        fclose(fp);
        return NULL;
    }

    uint32_t dib_size =
        (uint32_t)dib[0] |
        ((uint32_t)dib[1] << 8) |
        ((uint32_t)dib[2] << 16) |
        ((uint32_t)dib[3] << 24);

    if (dib_size < 40) {
        fprintf(stderr, "[ERROR] Unsupported BMP header: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    int32_t width =
        (int32_t)((uint32_t)dib[4] |
        ((uint32_t)dib[5] << 8) |
        ((uint32_t)dib[6] << 16) |
        ((uint32_t)dib[7] << 24));

    int32_t height =
        (int32_t)((uint32_t)dib[8] |
        ((uint32_t)dib[9] << 8) |
        ((uint32_t)dib[10] << 16) |
        ((uint32_t)dib[11] << 24));

    uint16_t planes =
        (uint16_t)dib[12] |
        ((uint16_t)dib[13] << 8);

    uint16_t bits =
        (uint16_t)dib[14] |
        ((uint16_t)dib[15] << 8);

    uint32_t compression =
        (uint32_t)dib[16] |
        ((uint32_t)dib[17] << 8) |
        ((uint32_t)dib[18] << 16) |
        ((uint32_t)dib[19] << 24);

    /*
     * Supported:
     *   8-bit grayscale/paletted BMP
     *   24-bit RGB BMP
     *   32-bit RGBA BMP
     */
    if (width <= 0 ||
        height == 0 ||
        planes != 1 ||
        (bits != 8 && bits != 24 && bits != 32) ||
        compression != 0) {

        fprintf(stderr,
                "[ERROR] BMP must be uncompressed 8-bit, 24-bit or 32-bit: %s\n",
                filename);

        fclose(fp);
        return NULL;
    }

    int top_down = (height < 0);
    int abs_height = height < 0 ? -height : height;

    /*
     * 8-bit BMP has a color palette.
     * The usual palette contains 256 entries,
     * each consisting of B, G, R, reserved.
     */
    unsigned char palette[256][4];

    if (bits == 8) {

        /*
         * A standard BITMAPINFOHEADER normally has 256 colors
         * when biClrUsed == 0 for an 8-bit image.
         */
        uint32_t colors_used =
            (uint32_t)dib[32] |
            ((uint32_t)dib[33] << 8) |
            ((uint32_t)dib[34] << 16) |
            ((uint32_t)dib[35] << 24);

        if (colors_used == 0)
            colors_used = 256;

        if (colors_used > 256)
            colors_used = 256;

        if (fread(palette, 4, colors_used, fp) != colors_used) {
            fprintf(stderr, "[ERROR] Cannot read BMP color palette: %s\n",
                    filename);

            fclose(fp);
            return NULL;
        }
    }

    int bytes_per_pixel = bits / 8;

    /*
     * BMP rows are padded to a multiple of 4 bytes.
     */
    int row_stride =
        ((width * bytes_per_pixel + 3) / 4) * 4;

    unsigned char *row =
        malloc((size_t)row_stride);

    float *out =
        malloc(28 * 28 * sizeof(float));

    if (row == NULL || out == NULL) {

        free(row);
        free(out);
        fclose(fp);

        return NULL;
    }

    /*
     * Convert the source image to 28x28.
     */
    for (int y = 0; y < 28; ++y) {

        int sy =
            (y * abs_height) / 28;

        if (sy >= abs_height)
            sy = abs_height - 1;

        int file_y =
            top_down
            ? sy
            : (abs_height - 1 - sy);

        if (fseek(
                fp,
                (long)pixel_offset +
                (long)file_y * row_stride,
                SEEK_SET) != 0 ||

            fread(
                row,
                1,
                row_stride,
                fp) != (size_t)row_stride) {

            free(row);
            free(out);
            fclose(fp);

            return NULL;
        }

        for (int x = 0; x < 28; ++x) {

            int sx =
                (x * width) / 28;

            if (sx >= width)
                sx = width - 1;

            float gray;

            if (bits == 8) {

                /*
                 * 8-bit BMP:
                 * pixel value is an index into the palette.
                 */
                unsigned char index =
                    row[sx];

                float b =
                    palette[index][0] / 255.0f;

                float g =
                    palette[index][1] / 255.0f;

                float r =
                    palette[index][2] / 255.0f;

                gray =
                    0.299f * r +
                    0.587f * g +
                    0.114f * b;

            } else {

                /*
                 * 24/32-bit BMP:
                 * BMP stores pixels as BGR(A).
                 */
                unsigned char *p =
                    row + sx * bytes_per_pixel;

                float b =
                    p[0] / 255.0f;

                float g =
                    p[1] / 255.0f;

                float r =
                    p[2] / 255.0f;

                gray =
                    0.299f * r +
                    0.587f * g +
                    0.114f * b;
            }

            out[y * 28 + x] = gray;
        }
    }

    free(row);
    fclose(fp);

    return out;
}

#endif /* MNIST_LOADER_H */
