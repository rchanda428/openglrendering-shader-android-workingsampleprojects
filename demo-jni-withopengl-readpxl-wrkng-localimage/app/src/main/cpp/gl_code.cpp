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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/ocl.hpp"

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


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_map);

    cv::Mat inputInMat = cv::imread("/storage/emulated/0/opencvTesting/wp4473260.jpg",cv::IMREAD_COLOR);
    if(inputInMat.empty())
        LOGI("input matImage is empty");

    LOGI("width:[%d], height:[%d], Channels[%d] \n", inputInMat.cols,inputInMat.rows, inputInMat.channels());

    int bw = inputInMat.cols;
    int bh = inputInMat.rows;

//    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, bw, bh, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, inputInMat.data); //this is for grey input image
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bw, bh, 0, GL_RGB, GL_UNSIGNED_BYTE, inputInMat.data);

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
    //dump output
    uint32_t* readPixels;
    readPixels = new uint32_t[bw * bh];
    glReadPixels(0, 0, bw, bh, GL_RGBA, GL_UNSIGNED_BYTE, readPixels);
    cv::Mat outputReadpixelInMat = cv::Mat(inputInMat.rows,inputInMat.cols,CV_8UC4,readPixels);
    if(outputReadpixelInMat.empty())
        LOGI("outputReadpixelInMat inputInMat empty");

    cv::imwrite("/storage/emulated/0/opencvTesting/outputReadpixelInMat.jpg", outputReadpixelInMat);
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