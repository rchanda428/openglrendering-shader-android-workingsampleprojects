/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenGL ES 2.0 code

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "cl_code.h"
#include "speckle_utils.h"
#define  LOG_TAG    "libgl2jni"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

namespace abc {}

static void printGLString(const char *name, GLenum s) {
    const char *v = (const char *) glGetString(s);
    LOGI("GL %s = %s\n", name, v);
}

static void checkGlError(const char *op) {
    for (GLint error = glGetError(); error; error = glGetError()) {
        LOGI("after %s() glError (0x%x)\n", op, error);
    }
}

inline unsigned char saturate_cast_uchar(float val) {
    //val += 0.5; // to round the value
    return static_cast<unsigned char>(val < 0 ? 0 : (val > 0xff ? 0xff : val));
}

void initializeBuffer(cl_command_queue &cmd_queue, cl_mem &buffer,
                      const size_t &buf_size, float initial_value)
{
    cl_int           err             = CL_SUCCESS;
    float *buffer_ptr = static_cast<float *>(clEnqueueMapBuffer(
            cmd_queue,
            buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            buf_size,
            0,
            NULL,
            NULL,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        DPRINTF("Error imageCounts_buffer. %d ", err);
        std::exit(err);
    }
    clFinish(cmd_queue);
    for (size_t i = 0; i < buf_size; ++i)
    {
        buffer_ptr[i] = initial_value;
    }

    err = clEnqueueUnmapMemObject(cmd_queue, buffer, buffer_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        DPRINTF("Error  unmapping imageCounts_buffer. %d", err);
        std::exit(err);
    }
}

template<typename type>
void dump_buffer_GPU_space(cl_command_queue& command_queue, cl_mem& buffer,
                           const std::string& filename, const size_t buf_size)
{
    cl_int err;
    err = CL_SUCCESS;
    type* mapped_ptr = static_cast<type*>(clEnqueueMapBuffer( command_queue, buffer, CL_TRUE, CL_MAP_READ,
                                                              0, buf_size * sizeof(type), 0, NULL, NULL, &err ));
    if (err != CL_SUCCESS)
    {
        DPRINTF("Error mapping output buffer. %d ",err);
        std::exit(err);
    }
    clFinish(command_queue);

    std::ofstream fout(filename, std::ios::binary);
    if (!fout)
    {
        DPRINTF("Couldn't open file %s ", filename);
        std::exit(EXIT_FAILURE);
    }

    char* output_image_U8 = new char[buf_size];

    if (sizeof(type) == sizeof(float))
    {
        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = saturate_cast_uchar(mapped_ptr[pix_i]);
    }
    else if (sizeof(type) == sizeof(char))
    {
        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = (unsigned char)mapped_ptr[pix_i];
    }
    fout.write(output_image_U8, buf_size);
    fout.close();
    delete[] output_image_U8;

    err = clEnqueueUnmapMemObject(command_queue, buffer, mapped_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        DPRINTF(" unmapping output buffer %d",err);
        std::exit(err);
    }
}

template<typename type>
void dump_buffer_GPU_space(cl_command_queue& command_queue, cl_mem& buffer,
                           std::ofstream &fout, const size_t buf_size)
{
    cl_int err;
    err = CL_SUCCESS;
    type* mapped_ptr = static_cast<type*>(clEnqueueMapBuffer( command_queue, buffer, CL_TRUE, CL_MAP_READ,
                                                              0, buf_size * sizeof(type), 0, NULL, NULL, &err ));
    if (err != CL_SUCCESS)
    {
        DPRINTF("Error mapping output buffer. %d ",err);
        std::exit(err);
    }
    clFinish(command_queue);

    char* output_image_U8 = new char[buf_size];

    if (sizeof(type) == sizeof(float))
    {
        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = saturate_cast_uchar(mapped_ptr[pix_i]);
    }
    else if (sizeof(type) == sizeof(char))
    {
        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = (unsigned char)mapped_ptr[pix_i];
    }
    fout.write(output_image_U8, buf_size);
    delete[] output_image_U8;

    err = clEnqueueUnmapMemObject(command_queue, buffer, mapped_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        DPRINTF(" unmapping output buffer %d",err);
        std::exit(err);
    }
}

static const char* PROGRAM_IMAGE_2D_COPY_SOURCE[] = {
        "__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE   | CLK_FILTER_NEAREST;\n",
        "__kernel void image2dCopy(__read_only image2d_t input, __write_only image2d_t output )\n",
        "{\n",
        "    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n",
        "    uint4 temp = read_imageui(input, imageSampler, coord);\n",
        "    write_imageui(output, coord, temp);\n",
        "}"
};

static const char *PROGRAM_COUNTS_SOURCE[] = {
        "__kernel void preblur(__global unsigned char *src,\n",
        //"                   __read_write image2d_t  imageCounts,\n",
        "                   __global float *imageCounts,\n",
        "                   __global float *imageSquaredCounts,\n",
        "                   __global float *lascaCounts,\n",
        "                   __global float *lascaSquaredCounts,\n",
        "                    float alpha,\n",
        "                    float one_minus_alpha,\n",
        "                    float beta,\n",
        "                    float one_minus_beta\n",
        "                   ){\n",
        "    uint wid_x = get_global_id(0);\n",
        //"    uint id_y = get_global_id(1);\n",
        //"    uint wid_x = (id_y * 1440) + (id_x);\n",
        //"    float squarebuffer = src[wid_x]*src[wid_x];\n",
        "    float squarebuffer = native_powr(src[wid_x],2);\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        /*"    wid_x = ((id_y+1) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+2) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+3) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+4) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+5) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+6) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",
        "    wid_x = ((id_y+7) * 1440) + (id_x);\n",
        "    squarebuffer = src[wid_x]*src[wid_x];\n",
        "    imageCounts[wid_x] = one_minus_alpha*src[wid_x] + alpha*imageCounts[wid_x];\n",
        "    imageSquaredCounts[wid_x] = one_minus_alpha*squarebuffer + alpha*imageSquaredCounts[wid_x];\n",
        "    lascaCounts[wid_x] = one_minus_beta*imageCounts[wid_x]+ beta*lascaCounts[wid_x];\n",
        "    lascaSquaredCounts[wid_x] = one_minus_beta*imageSquaredCounts[wid_x] + beta*lascaSquaredCounts[wid_x];\n",*/
        "}\n"
};

static const char* PROGRAM_BLUR_SOURCE[] = {
        "__kernel void blur(__global float *lascaCounts,\n",
        "                   __global float *lascaSquaredCounts,\n",
        "                   __global float *lascaCountsBlur,\n",
        "                   __global float *lascaSquaredCountsBlur,\n",
        //"                   __global unsigned char *imageContrast,\n",
        "                   __global unsigned char *nanMask,\n",
        "                   __global unsigned char *turbo,\n",
        "                    float contrast\n",
        "                   ){\n",
        "    uint wid_x = get_global_id(0);\n",
        "    lascaCountsBlur[wid_x] = 0.111111*(lascaCounts[wid_x]+lascaCounts[wid_x+1]+lascaCounts[wid_x-1]+lascaCounts[wid_x+1440]+lascaCounts[wid_x+1441]+lascaCounts[wid_x+1439]+lascaCounts[wid_x-1440]+lascaCounts[wid_x-1441]+lascaCounts[wid_x-1439]);\n",
        "    lascaSquaredCountsBlur[wid_x] = 0.111111*(lascaSquaredCounts[wid_x]+lascaSquaredCounts[wid_x+1]+lascaSquaredCounts[wid_x-1]+lascaSquaredCounts[wid_x+1440]+lascaSquaredCounts[wid_x+1441]+lascaSquaredCounts[wid_x+1439]+lascaSquaredCounts[wid_x-1440]+lascaSquaredCounts[wid_x-1441]+lascaSquaredCounts[wid_x-1439]);\n",
        "    if(lascaCountsBlur[wid_x] < 5 ) {\n",
        "       nanMask[wid_x] = 0;\n",
        //"       wid_x = wid_x*3;\n",
        //"       imageContrast[wid_x++] = 255;\n",
        //"       imageContrast[wid_x++] = 255;\n",
        //"       imageContrast[wid_x] = 255;\n",
        "    } else if ( lascaCountsBlur[wid_x] > 250) {\n",
        "       nanMask[wid_x] = 0;\n",
        "    } else {\n",
        //"       float lascaBuffer = lascaCountsBlur[wid_x]*lascaCountsBlur[wid_x];\n",
        "       float lascaBuffer = native_powr(lascaCountsBlur[wid_x],2);\n",
        "       float tempfloat  = contrast * native_divide((lascaSquaredCountsBlur[wid_x] - lascaBuffer),lascaBuffer);\n",
        //"       uint temp = select(select(tempfloat, 255.0f, isgreater(tempfloat,255.0f)), 0.0f, isgreater(0.0f,tempfloat));\n",
        //"       uchar temp = ;\n",
        //"       nanMask[wid_x] = 255-convert_uchar_sat_rte(tempfloat);\n",// gray scale image value
        "       nanMask[wid_x] = convert_uchar_sat_rte(tempfloat);\n",// gray scale image value
        "    }\n",
        "}\n"
};//should change to 2,1,0

std::vector<char> image_buf(1555200);
float bw, bh;
void *img;

static float _alpha;
static float _beta;
static float _one_minus_alpha;
static float _one_minus_beta;
static float _contrast;

cl_wrapper       wrapper;
static cl_uint PROGRAM_IMAGE_2D_COPY_SOURCE_LEN;
cl_context       context;
cl_command_queue command_queue;
cl_program       program_image_2d_program;
cl_kernel        program_image_2d_kernel;

cl_int status = CL_SUCCESS;
// Create and initialize image objects
cl_image_desc imageDesc;
cl_image_format imageFormat;
cl_mem inputImage2D;
cl_mem outputImage2D;
unsigned char *outputImageData2D;
char *output_image_U8;
void init_cl()
{
    PROGRAM_IMAGE_2D_COPY_SOURCE_LEN = sizeof(PROGRAM_IMAGE_2D_COPY_SOURCE) / sizeof(const char*);
    context = wrapper.get_context();
    command_queue = wrapper.get_command_queue();
    program_image_2d_program = wrapper.make_program(PROGRAM_IMAGE_2D_COPY_SOURCE, PROGRAM_IMAGE_2D_COPY_SOURCE_LEN);
    program_image_2d_kernel = wrapper.make_kernel("image2dCopy", program_image_2d_program);

    memset(&imageDesc, '\0', sizeof(cl_image_desc));
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = 1440;
    imageDesc.image_height = 1080;

    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    imageFormat.image_channel_order = CL_R;

    std::string src_filename = std::string("/storage/emulated/0/opencvTesting/out148Yonly.yuv");
    std::ifstream fin(src_filename, std::ios::binary);
    if (!fin) {
        DPRINTF("Couldn't open file %s", src_filename.c_str());
        std::exit(EXIT_FAILURE);
    }
    const auto        fin_begin = fin.tellg();
    fin.seekg(0, std::ios::end);
    const auto        fin_end = fin.tellg();
    const size_t      buf_size = static_cast<size_t>(fin_end - fin_begin);
    fin.seekg(0, std::ios::beg);
    fin.read(image_buf.data(), buf_size);
    bw = 1440;
    bh = 1080;
    outputImageData2D = (unsigned char*)malloc(1440* 1080);
    inputImage2D = clCreateImage(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 &imageFormat,
                                 &imageDesc,
                                 image_buf.data(),
                                 &status);

    // Create 2D output image
    outputImage2D = clCreateImage(context,
                                  CL_MEM_WRITE_ONLY,
                                  &imageFormat,
                                  &imageDesc,
                                  0,
                                  &status);
}

void execute_cl()
{
    cl_mem objs[] = { outputImage2D };
    status = clSetKernelArg(program_image_2d_kernel,0,sizeof(cl_mem),&inputImage2D);
    status = clSetKernelArg(program_image_2d_kernel,1,sizeof(cl_mem),&outputImage2D);
    size_t globalThreads[] = { 1440, 1080 };

    struct timeval startNDKKernel, endNDKKernel;
    gettimeofday(&startNDKKernel, NULL);
    status = clEnqueueNDRangeKernel( command_queue, program_image_2d_kernel, 2, NULL, globalThreads, 0, 0, NULL,0);
    clFinish(command_queue);
    gettimeofday(&endNDKKernel, NULL);
    DPRINTF("time taken :%ld",((endNDKKernel.tv_sec * 1000000 + endNDKKernel.tv_usec)- (startNDKKernel.tv_sec * 1000000 + startNDKKernel.tv_usec)));
#if OPENGL
    // Enqueue Read Image
    static int count = 0;
    if( count > 50 && count < 70 ) {
        size_t origin[] = { 0, 0, 0 };
        size_t region[] = { 1440, 1080, 1 };

        unsigned char *outputImageData2D = (unsigned char*)malloc(1440* 1080);
        // Read output of 2D copy
        clEnqueueAcquireGLObjects(command_queue, 1, objs, 0, NULL, NULL);
        status = clEnqueueReadImage(command_queue, outputImage2D, 1,
                                origin, region, 0, 0, outputImageData2D, 0, 0, 0);
        clEnqueueReleaseGLObjects(command_queue, 1, objs, 0, NULL, NULL);

        std::string filename("/storage/emulated/0/opencvTesting/output_copy" + std::to_string(count)+".yuv");
        std::ofstream fout(filename, std::ios::binary);
        if (!fout) {
            std::cerr << "Couldn't open file " << filename << "\n";
            std::exit(EXIT_FAILURE);
        }

        const size_t buf_size = imageDesc.image_height * imageDesc.image_width;
        char *output_image_U8 = new char[buf_size];

        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = (unsigned char) outputImageData2D[pix_i];
        fout.write(output_image_U8, buf_size);
        fout.close();
        delete[] output_image_U8;
        free(outputImageData2D);

    }
    count++;
#endif
#if 1
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { 1440, 1080, 1 };
    // Read output of 2D copy
    status = clEnqueueReadImage(command_queue, outputImage2D, 1, origin, region, 0, 0, outputImageData2D, 0, 0, 0);
    clFinish(command_queue);
    static int count_ = 0;
    if(count_ == 0) {
        std::string filename(
                "/storage/emulated/0/opencvTesting/output_copy.yuv");
        std::ofstream fout(filename, std::ios::binary);
        if (!fout) {
            std::cerr << "Couldn't open file " << filename << "\n";
            std::exit(EXIT_FAILURE);
        }
        const size_t buf_size = imageDesc.image_height * imageDesc.image_width;
        output_image_U8 = new char[buf_size];

        for (int pix_i = 0; pix_i < buf_size; pix_i++)
            output_image_U8[pix_i] = (unsigned char) outputImageData2D[pix_i];
        fout.write(output_image_U8, buf_size);
        fout.close();
        delete[] output_image_U8;
        count_++;
    }
    //free(outputImageData2D);
#endif
}

unsigned char * rawData = NULL;
FILE *fp = NULL;
size_t  buf_size;

std::ofstream fout;
struct timeval startNDKKernel, endNDKKernel;
size_t global_preblur[] = {1440, 1080};
size_t global_size = 1552317;
size_t offset = 1441;

cl_int err = CL_SUCCESS;

cl_program       preblur_program;
cl_kernel        preblur_kernel;
cl_program       blur_program;
cl_kernel        blur_kernel;

cl_mem turbo_buffer;
cl_mem src_buffer = nullptr;
cl_mem imageCounts_buffer;
//cl_mem imageCounts_Image2D;
cl_mem imageSquaredCounts_buffer;
cl_mem lascaCounts_buffer;
cl_mem lascaSquaredCounts_buffer;
cl_mem lascaCountsBlur_buffer;
cl_mem lascaSquaredCountsBlur_buffer;
//cl_mem imageContrast_buffer;
cl_mem nanMask_buffer;

cl_mem temp_buffer;

cl_event k_events_k1[1];
cl_event k_events_k2[1];

void set_parameters()
{
    _alpha = 0.93f;
    _beta = 0.75f;
    _one_minus_alpha = 1.0f - _alpha;
    _one_minus_beta = 1.0f - _beta;
    _contrast = 2600.0f;
}
void setKernelArguments();
void speckle_init()
{
    DPRINTF("read input file");
    std::string FileName("/storage/emulated/0/opencvTesting/tina60-120");
    fp = fopen(FileName.c_str(),"r+b");
    if(NULL == fp) {
        DPRINTF(" fopen() Error!!!\n");
    }
    bw = 1440;
    bh = 1080 ;
    buf_size = bw*bh;
    //Allocate Buffer for rawData
    rawData = (unsigned char *)malloc(buf_size);
    if (NULL == rawData) {
        DPRINTF("Rawdata is NULL\n");
    }

    /*std::string Filename = "/storage/emulated/0/opencvTesting/SpeckleOutputCL_new.raw";
    fout = std::ofstream(Filename, std::ios::binary);
    if (!fout) {
        printf("Cannot save the video to a SpeckleOutputCL.raw file");
        std::exit(EXIT_FAILURE);
    }*/

    outputImageData2D = (unsigned char*)malloc(1440* 1080);

    static const cl_uint PROGRAM_COUNTS_SOURCE_LEN = sizeof(PROGRAM_COUNTS_SOURCE) / sizeof(const char*);
    static const cl_uint PROGRAM_BLUR_SOURCE_LEN = sizeof(PROGRAM_BLUR_SOURCE) / sizeof(const char*);

    context = wrapper.get_context();
    command_queue = wrapper.get_command_queue();
    preblur_program = wrapper.make_program(PROGRAM_COUNTS_SOURCE, PROGRAM_COUNTS_SOURCE_LEN);
    preblur_kernel = wrapper.make_kernel("preblur", preblur_program);
    blur_program = wrapper.make_program(PROGRAM_BLUR_SOURCE, PROGRAM_BLUR_SOURCE_LEN);
    blur_kernel = wrapper.make_kernel("blur", blur_program);

    turbo_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 768, turbo_array, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for turbo_buffer %d",err);
        std::exit(err);
    }

    imageCounts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for imageCounts_buffer %d",err);
        std::exit(err);
    }
    initializeBuffer(command_queue, imageCounts_buffer, buf_size, 0.0f);

    imageSquaredCounts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for imageSquaredCounts_buffer %d",err);
        std::exit(err);
    }
    initializeBuffer(command_queue, imageSquaredCounts_buffer, buf_size, 0.0f);
    lascaCounts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for lascaCounts_buffer %d",err);
        std::exit(err);
    }
    initializeBuffer(command_queue, lascaCounts_buffer, buf_size, 0.0f);
    lascaSquaredCounts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for lascaSquaredCounts_buffer %d",err);
        std::exit(err);
    }
    initializeBuffer(command_queue, lascaSquaredCounts_buffer, buf_size, 0.0f);
    lascaCountsBlur_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for lascaCountsBlur_buffer %d",err);
        std::exit(err);
    }
    lascaSquaredCountsBlur_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for lascaSquaredCountsBlur_buffer %d",err);
        std::exit(err);
    }
    /*imageContrast_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(char) * 3, NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for imageContrast_buffer %d",err);
        std::exit(err);
    }*/
    nanMask_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size * sizeof(char), NULL, &err);
    if (err != CL_SUCCESS) {
        DPRINTF("clCreateBuffer for nanMask_buffer %d",err);
        std::exit(err);
    }
}

void setKernelArguments()
{
    /*
         * Step 1: Set up pre-blur kernel arguments and run the pre-blur kernel.
         */
    err = clSetKernelArg(preblur_kernel, 1, sizeof(imageCounts_buffer), &imageCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 1 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 2, sizeof(imageSquaredCounts_buffer),
                         &imageSquaredCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 2 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 3, sizeof(lascaCounts_buffer), &lascaCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 3 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 4, sizeof(lascaSquaredCounts_buffer),
                         &lascaSquaredCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 4 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 5, sizeof(_alpha), &_alpha);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 5 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 6, sizeof(_one_minus_alpha), &_one_minus_alpha);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 6 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 7, sizeof(_beta), &_beta);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 7 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(preblur_kernel, 8, sizeof(_one_minus_beta), &_one_minus_beta);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 8 with error %d", err);
        std::exit(err);
    }

    err = clSetKernelArg(blur_kernel, 0, sizeof(lascaCounts_buffer), &lascaCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 0 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(blur_kernel, 1, sizeof(lascaSquaredCounts_buffer),
                         &lascaSquaredCounts_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 1 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(blur_kernel, 2, sizeof(lascaCountsBlur_buffer),
                         &lascaCountsBlur_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 2 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(blur_kernel, 3, sizeof(lascaSquaredCountsBlur_buffer),
                         &lascaSquaredCountsBlur_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 3 with error %d", err);
        std::exit(err);
    }
    /*err = clSetKernelArg(blur_kernel, 4, sizeof(imageContrast_buffer), &imageContrast_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 4 with error %d", err);
        std::exit(err);
    }*/
    err = clSetKernelArg(blur_kernel, 4, sizeof(nanMask_buffer), &nanMask_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 5 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(blur_kernel, 5, sizeof(turbo_buffer), &turbo_buffer);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 6 with error %d", err);
        std::exit(err);
    }
    err = clSetKernelArg(blur_kernel, 6, sizeof(_contrast), &_contrast);
    if (err != CL_SUCCESS) {
        DPRINTF("clSetKernelArg for argument 7 with error %d", err);
        std::exit(err);
    }
}

void speckle_execute() {
    if (!feof(fp)) {
        fread(rawData, buf_size, 1, fp);
        static int c = 0;
        //DPRINTF("processing frame number %d", c);

        // read a new frame from video
        //bool bSuccess = cap.read(frame);
        //DPRINTF("Process frame started in CL");
        gettimeofday(&startNDKKernel, NULL);

        src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, buf_size, rawData, &err);
        /*if (err != CL_SUCCESS) {
            DPRINTF("clCreateBuffer for src_buffer %d ", err);
            std::exit(err);
        }*/

        //setKernelArguments();
        err = clSetKernelArg(preblur_kernel, 0, sizeof(src_buffer), &src_buffer);
        /*if (err != CL_SUCCESS) {
            DPRINTF("clSetKernelArg for argument 0 with error %d", err);
            std::exit(err);
        }*/

        /******* pre blur Kernel execution********************/
        //DPRINTF("start of pre-blur kernel");
        // global_blur[] = { 1338, 1078 };
        err = clEnqueueNDRangeKernel(command_queue, preblur_kernel, 1, &offset, &global_size, NULL, 0, NULL, &k_events_k1[0]);
        /*if (err != CL_SUCCESS) {
            DPRINTF("preblur_kernel clEnqueueNDRangeKernel with error %d", err);
            std::exit(err);
        }*/
        /******* blur Kernel execution********************/
        //DPRINTF("start of blur kernel");
        err = clEnqueueNDRangeKernel(command_queue, blur_kernel, 1, &offset, &global_size, NULL, 1, k_events_k1, NULL);
        /*if (err != CL_SUCCESS) {
            DPRINTF("blur_kernel clEnqueueNDRangeKernel with error %d", err);
            std::exit(err);
        }*/

        temp_buffer = lascaCounts_buffer;
        lascaCounts_buffer = lascaCountsBlur_buffer;
        lascaCountsBlur_buffer = temp_buffer;

        temp_buffer = lascaSquaredCounts_buffer;
        lascaSquaredCounts_buffer = lascaSquaredCountsBlur_buffer;
        lascaSquaredCountsBlur_buffer = temp_buffer;

        //dump_buffer_GPU_space<char>(command_queue, imageContrast_buffer, fout, buf_size * 3);
        status = clEnqueueReadBuffer(command_queue, nanMask_buffer, 1, NULL, buf_size, outputImageData2D, 0, 0, 0);
        clReleaseMemObject(src_buffer);
        clFinish(command_queue);
        gettimeofday(&endNDKKernel, NULL);
        DPRINTF("time taken frame %d :%ld", c,((endNDKKernel.tv_sec * 1000000 + endNDKKernel.tv_usec)-(startNDKKernel.tv_sec * 1000000 + startNDKKernel.tv_usec)));
        //DPRINTF("Process frame ended in CL");
        c++;
    }
    else
    {
        fclose(fp);
        fp = NULL;
        std::string FileName("/storage/emulated/0/opencvTesting/tina60-120");
        fp = fopen(FileName.c_str(),"r+b");
        if(NULL == fp) {
            DPRINTF(" fopen() Error!!!\n");
        }
        fout.close();

        initializeBuffer(command_queue, imageCounts_buffer, buf_size, 0.0f);
        initializeBuffer(command_queue, imageSquaredCounts_buffer, buf_size, 0.0f);
        initializeBuffer(command_queue, lascaCounts_buffer, buf_size, 0.0f);
        initializeBuffer(command_queue, lascaSquaredCounts_buffer, buf_size, 0.0f);
        //clReleaseMemObject(imageCounts_Image2D);
        /*clReleaseMemObject(nanMask_buffer);
        //clReleaseMemObject(imageContrast_buffer);
        clReleaseMemObject(imageCounts_buffer);
        clReleaseMemObject(imageSquaredCounts_buffer);
        clReleaseMemObject(lascaCountsBlur_buffer);
        clReleaseMemObject(lascaSquaredCountsBlur_buffer);
        clReleaseMemObject(lascaCounts_buffer);
        clReleaseMemObject(lascaSquaredCounts_buffer);*/
    }
}

auto gVertexShader =
        "attribute vec4 aPosition;\n"
            "attribute vec2 aTexCoord;\n"
            "varying vec2 vTexCoord;\n"
            "void main() {\n"
            "   gl_Position = vec4(aPosition, 0.0, 1.0);\n"
            "   vTexCoord = aTexCoord;\n"
            "}";

auto gFragmentShader =
        "precision mediump float;\n"
            "uniform sampler2D rubyTexture;\n"
            "varying vec2 vTexCoord;\n"
            "void main() {\n"
            "   vec4 color = texture2D(rubyTexture, vTexCoord);\n"
            "   gl_FragColor = color;\n"
            "}";


GLuint loadShader(GLenum shaderType, const char *pSource) {
    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        glShaderSource(shader, 1, &pSource, NULL);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                char *buf = (char *) malloc(infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, NULL, buf);
                    LOGE("Could not compile shader %d:\n%s\n",
                         shaderType, buf);
                    free(buf);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }

    return shader;
}

GLuint createProgram(const char *pVertexSource, const char *pFragmentSource) {
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, pVertexSource);
    if (!vertexShader) {
        return 0;
    }

    GLuint pixelShader = loadShader(GL_FRAGMENT_SHADER, pFragmentSource);
    if (!pixelShader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, vertexShader);
        checkGlError("glAttachShader");
        glAttachShader(program, pixelShader);
        checkGlError("glAttachShader");
        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char *buf = (char *) malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    LOGE("Could not link program:\n%s\n", buf);
                    free(buf);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

GLuint programId;
GLuint aPosition;
GLuint aTexCoord;
GLuint rubyTexture;
GLuint lut;
GLuint rubyTextureSize;
GLuint rubyInputSize;
GLuint rubyOutputSize;

GLuint texture_map;
GLuint lut_map;

int scnw, scnh, vw, vh;
char *gVs, *gFs;

bool initProgram() {
//    LOGI("initProgram vs=%s fs=%s", gVs, gFs);
    if(!gVs)
        gVs = (char *)gVertexShader;
    if(!gFs)
        gFs = (char *)gFragmentShader;
    programId = createProgram(gVs, gFs);
    if (!programId) {
        LOGE("Could not create program.");
        return false;
    }

    aPosition = glGetAttribLocation(programId, "aPosition");
    aTexCoord = glGetAttribLocation(programId, "aTexCoord");
    rubyTexture = glGetUniformLocation(programId, "rubyTexture");
    lut = glGetUniformLocation(programId, "lut");
    rubyTextureSize = glGetUniformLocation(programId, "rubyTextureSize");
    rubyInputSize = glGetUniformLocation(programId, "rubyInputSize");
    rubyOutputSize = glGetUniformLocation(programId, "rubyOutputSize");

    return true;
}

bool setupGraphics(int w, int h) {
    scnw = w;
    scnh = h;
    LOGI("setupGraphics(%d, %d)", w, h);
    glViewport(0, 0, scnw, scnh);
    return true;
}

const GLfloat gTriangleVertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        -1.0f, 1.0f,

        -1.0f, 1.0f,
        1.0f,-1.0f,
        1.0f,1.0f
};

const GLfloat gTexVertices[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,

        0.0f, 0.0f,
        1.0f,1.0f,
        1.0f,0.0f
};

void renderFrame() // 16.6ms
{
    float grey;
    grey = 0.00f;

    glClearColor(grey, grey, grey, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    //execute_cl();
    speckle_execute();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_map);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, bw, bh, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, outputImageData2D);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bw, bh, 0, GL_RGB, GL_UNSIGNED_BYTE, outputImageData2D);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lut_map);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, reverse_turbo_array_1);

    glUseProgram(programId);

    glVertexAttribPointer(aPosition, 2, GL_FLOAT, GL_FALSE, 0, gTriangleVertices);
    glEnableVertexAttribArray(aPosition);

    glVertexAttribPointer(aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, gTexVertices);
    glEnableVertexAttribArray(aTexCoord);

    glUniform2f(rubyTextureSize, bw, bh);
    glUniform1i(rubyTexture, 0);
    glUniform1i(lut, 1);

    //if(rubyInputSize >= 0)
    //    glUniform2f(rubyInputSize, bw, bh);
    //if(rubyOutputSize >= 0)
    //    glUniform2f(rubyOutputSize, vw, vh);

//    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    static int i = 0;
    /*
    if ( i == 20) {
        GLubyte *data = (GLubyte *) malloc(4 * scnw*scnh);
        if (data) {
            glReadPixels(0, 0, scnw, scnh, GL_RGBA, GL_UNSIGNED_BYTE, data);;
        }
        std::string filename("/storage/emulated/0/opencvTesting/image.rgb");
        std::ofstream fout(filename, std::ios::binary);
        fout.write((char *) data, 4 * scnw*scnh);
        fout.close();
        free(data);
    }*/
    i++;
}

void _init(JNIEnv *env, jobject bmp) {
    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);

    //AndroidBitmapInfo info;
    //AndroidBitmap_getInfo(env, bmp, &info);
    //bw = info.width;
    //bh = info.height;
    //AndroidBitmap_lockPixels(env, bmp, &img);

    glGenTextures(1, &texture_map);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_map);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &lut_map);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lut_map);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

}

void _loadShader(JNIEnv *env, jstring jvs, jstring jfs) {
    jboolean b;
    if(jvs) {
        const char *vs = env->GetStringUTFChars(jvs, &b);
        gVs = vs? strdup(vs) : NULL;
        env->ReleaseStringUTFChars(jvs, vs);
    } else
        gVs = (char *)gVertexShader;

    if(jfs) {
        const char *fs = env->GetStringUTFChars(jfs, &b);
        gFs = fs? strdup(fs) : NULL;
        env->ReleaseStringUTFChars(jfs, fs);
    } else
        gFs = (char *)gFragmentShader;

    initProgram();
}

extern "C" {
JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_init(JNIEnv *env, jobject obj, jobject bmp)
{
    //testOpencl();
    //test_cl_image();
    //init_cl();
    set_parameters();
    speckle_init();
    setKernelArguments();
    _init(env, bmp);
}

JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_resize(JNIEnv *env, jobject obj, jint width, jint height)
{
    setupGraphics(width, height);
}

JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_step(JNIEnv *env, jobject obj)
{
    renderFrame();
}

JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_loadShader(JNIEnv *env, jobject obj, jstring vs, jstring fs)
{
    _loadShader(env, vs, fs);
}
};