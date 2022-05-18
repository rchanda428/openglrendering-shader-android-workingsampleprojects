package com.android.gl2jni;
import static android.opengl.GLES20.GL_BUFFER_SIZE;
import static android.opengl.GLES20.GL_BUFFER_USAGE;
import static android.opengl.GLES20.GL_MAX_RENDERBUFFER_SIZE;
import static android.opengl.GLES20.GL_MAX_TEXTURE_IMAGE_UNITS;
import static android.opengl.GLES20.GL_MAX_TEXTURE_SIZE;
import static android.opengl.GLES20.glGetIntegerv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class GLMergerWithShader {
    public final static String TAG = "GLMergerWithShader";
    Mat outputReadpixelInMat;
    Mat mat_RGBA2YUV_I420;
    Mat testInputImgJPEGmat;
    Mat testConvertedImgYUVmat;
    private int mScreenWidth  = 1920;
    private int mScreenHeight = 1080;
    private int mProgramId;
    private int aPosition;
    private int aTexCoord;
    private int rubyTexture1Y;
    private int rubyTexture1U;
    private int rubyTexture1V;
    private int rubyTexture2Y;
    private int rubyTexture2U;
    private int rubyTexture2V;
    private int rubyTexture3Y;
    private int rubyTexture3U;
    private int rubyTexture3V;
    private int rubyTexture4Y;
    private int rubyTexture4U;
    private int rubyTexture4V;
    private int rubyTexture5;

    private int rubyTextureSize;
    private int texture_map1;
    private int texture_map2;
    private int texture_map3;
    private int texture_map4;
    private int texture_map5;
    private int texture_map6;
    private int texture_map7;
    private int texture_map8;
    private int texture_map9;
    private int texture_map10;
    private int texture_map11;
    private int texture_map12;
    private int texture_map13;
    private FloatBuffer mPosTriangleVertices;
    private FloatBuffer mTexVertices;
    private Context mAssetContext;
//    private ByteBuffer mglReadPixelBuf;                       // used by saveFrame
    private static final float[] gTriangleVertices = {
            -1.0f, 0.0f,
            0.0f, -0.0f,
            -1.0f, 1.0f,

            -1.0f, 1.0f,
            0.0f,-0.0f,
            0.0f,1.0f,
//1st img end
            -0.0f, -0.0f,
            1.0f, -0.0f,
            -0.0f, 1.0f,

            -0.0f, 1.0f,
            1.0f,-0.0f,
            1.0f,1.0f,
//2nd img end
            -1.0f, -1.0f,
            0.0f, -1.0f,
            -1.0f, 0.0f,

            -1.0f, 0.0f,
            0.0f,-1.0f,
            0.0f,0.0f,
//3rd image end
            -0.0f, -1.0f,
            1.0f, -1.0f,
            -0.0f, 0.0f,

            -0.0f, 0.0f,
            1.0f,-1.0f,
            1.0f,0.0f
//4th image end
    };

public static final float[] gTexVertices = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,

            0.0f, 0.0f,
            1.0f,1.0f,
            1.0f,0.0f,
//1st img end
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,

            0.0f, 0.0f,
            1.0f,1.0f,
            1.0f,0.0f,
//2nd img end
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,

            0.0f, 0.0f,
            1.0f,1.0f,
            1.0f,0.0f,
//3rd img end
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,

            0.0f, 0.0f,
            1.0f,1.0f,
            1.0f,0.0f
            //4th img end
    };

    private static final String MERGE_VERTEX_SHADER =
                    "attribute vec2 aPosition;\n" +
                    "attribute vec2 aTexCoord;\n" +
                    "varying vec2 vTexCoord;\n" +
                    "varying vec2 aPositionTexCoord;\n" +
                    "void main() {\n" +
                    "vTexCoord = aTexCoord;\n" +
                    "aPositionTexCoord = vec2(aPosition.x,-aPosition.y);\n" +
                    "gl_Position = vec4(aPosition.x,-aPosition.y, 0.0, 1.0);\n" +
                    "}\n";

    private static final String MERGE_FRAGMENT_SHADER =
                    "#ifdef GL_FRAGMENT_PRECISION_HIGH\n" +
                    "precision highp float;\n" +
                    "#else\n" +
                    "precision mediump float;\n" +
                    "#endif\n" +
                    "uniform sampler2D rubyTexture1Y;\n" +
                    "uniform sampler2D rubyTexture1U;\n" +
                    "uniform sampler2D rubyTexture1V;\n" +
                    "uniform sampler2D rubyTexture2Y;\n" +
                    "uniform sampler2D rubyTexture2U;\n" +
                    "uniform sampler2D rubyTexture2V;\n" +
                    "uniform sampler2D rubyTexture3Y;\n" +
                    "uniform sampler2D rubyTexture3U;\n" +
                    "uniform sampler2D rubyTexture3V;\n" +
                    "uniform sampler2D rubyTexture4Y;\n" +
                    "uniform sampler2D rubyTexture4U;\n" +
                    "uniform sampler2D rubyTexture4V;\n" +
                    "uniform sampler2D rubyTexture5;\n" +
                    "varying vec2 vTexCoord;\n" +
                    "varying vec2 aPositionTexCoord;\n" +
                    "void main() {\n" +
                            "float r1,g1,b1,y1,u1,v1;\n" +
                            "float r2,g2,b2,y2,u2,v2;\n" +
                            "float r3,g3,b3,y3,u3,v3;\n" +
                            "float r4,g4,b4,y4,u4,v4;\n" +
                            "y1 = texture2D(rubyTexture1Y, vTexCoord).r;\n" +
                            "u1 = texture2D(rubyTexture1U, vTexCoord).r;\n" +
                            "v1 = texture2D(rubyTexture1V, vTexCoord).r;\n"+
                            "y1 = 1.1643*(y1-0.0625);\n"+
                            "u1 = u1-0.5;\n" +
                            "v1 = v1-0.5;\n" +
                            "r1 = y1+1.5958*v1;\n" +
                            "g1 = y1-0.39173*u1-0.81290*v1;\n" +
                            "b1 = y1+2.017*u1;\n" +

                            "y2 = texture2D(rubyTexture2Y, vTexCoord).r;\n" +
                            "u2 = texture2D(rubyTexture2U, vTexCoord).r;\n" +
                            "v2 = texture2D(rubyTexture2V, vTexCoord).r;\n"+
                            "y2 = 1.1643*(y2-0.0625);\n"+
                            "u2 = u2-0.5;\n" +
                            "v2 = v2-0.5;\n" +
                            "r2 = y2+1.5958*v2;\n" +
                            "g2 = y2-0.39173*u2-0.81290*v2;\n" +
                            "b2 = y2+2.017*u2;\n" +

                            "y3 = texture2D(rubyTexture3Y, vTexCoord).r;\n" +
                            "u3 = texture2D(rubyTexture3U, vTexCoord).r;\n" +
                            "v3 = texture2D(rubyTexture3V, vTexCoord).r;\n"+
                            "y3 = 1.1643*(y3-0.0625);\n"+
                            "u3 = u3-0.5;\n" +
                            "v3 = v3-0.5;\n" +
                            "r3 = y3+1.5958*v3;\n" +
                            "g3 = y3-0.39173*u3-0.81290*v3;\n" +
                            "b3 = y3+2.017*u3;\n" +

                            "y4 = texture2D(rubyTexture4Y, vTexCoord).r;\n" +
                            "u4 = texture2D(rubyTexture4U, vTexCoord).r;\n" +
                            "v4 = texture2D(rubyTexture4V, vTexCoord).r;\n"+
                            "y4 = 1.1643*(y4-0.0625);\n"+
                            "u4 = u4-0.5;\n" +
                            "v4 = v4-0.5;\n" +
                            "r4 = y4+1.5958*v4;\n" +
                            "g4 = y4-0.39173*u4-0.81290*v4;\n" +
                            "b4 = y4+2.017*u4;\n" +

                            "if((aPositionTexCoord.x > 0.0)&&(aPositionTexCoord.y > 0.0))\n" +
                            "gl_FragColor = vec4(b1,g1,r1, 1.0);\n" +
                            "if((aPositionTexCoord.x > 0.0)&&(aPositionTexCoord.y < 0.0))\n" +
                            "gl_FragColor = vec4(b2,g2,r2, 1.0);\n" +
                            "if((aPositionTexCoord.x < 0.0)&&(aPositionTexCoord.y < 0.0))\n" +
                            "gl_FragColor = vec4(b3,g3,r3, 1.0);\n" +
                            "if((aPositionTexCoord.x < 0.0)&&(aPositionTexCoord.y > 0.0))\n" +
//                            "gl_FragColor = vec4(b4,g4,r4, 1.0);\n" +
//                            "gl_FragColor = vec4(b3,g3,r3, 1.0);\n" +
                            "gl_FragColor.bgra = texture2D(rubyTexture5, vTexCoord);\n" +
                    "}\n";

    private static final int FLOAT_SIZE_BYTES = 4;
    public GLMergerWithShader(){
        Log.d(TAG,"GLMergerWithShader entry");
        // Setup coordinate buffers
        mPosTriangleVertices = ByteBuffer.allocateDirect(gTriangleVertices.length*FLOAT_SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        mPosTriangleVertices.put(gTriangleVertices).position(0);
        mTexVertices = ByteBuffer.allocateDirect(gTexVertices.length * FLOAT_SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        mTexVertices.put(gTexVertices).position(0);
        Log.d(TAG,"GLMergerWithShader exit");
    }

    public void SetAssetContext(Context assetContext){
        Log.d(TAG,"SetAssetContext entry");
        mAssetContext = assetContext;
        Log.d(TAG,"SetAssetContext exit");
    }
    public static int loadShader(int shaderType, String source) {
        Log.d(TAG,"loadShader entry");
        int shader = GLES20.glCreateShader(shaderType);
        if (shader != 0) {
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            int[] compiled = new int[1];
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
            if (compiled[0] == 0) {
                String info = GLES20.glGetShaderInfoLog(shader);
                GLES20.glDeleteShader(shader);
                throw new RuntimeException("Could not compile shader " + shaderType + ":" + info);
            }
        }
        Log.d(TAG,"loadShader exit");
        return shader;
    }

    public static int createProgram(String vertexSource, String fragmentSource) {
        Log.d(TAG,"createProgram entry");
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        if (vertexShader == 0) {
            return 0;
        }
        int pixelShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (pixelShader == 0) {
            return 0;
        }

        int program = GLES20.glCreateProgram();
        if (program != 0) {
            GLES20.glAttachShader(program, vertexShader);
            checkGlError("glAttachShader");
            GLES20.glAttachShader(program, pixelShader);
            checkGlError("glAttachShader");
            GLES20.glLinkProgram(program);
            int[] linkStatus = new int[1];
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus,
                    0);
            if (linkStatus[0] != GLES20.GL_TRUE) {
                String info = GLES20.glGetProgramInfoLog(program);
                GLES20.glDeleteProgram(program);
                throw new RuntimeException("Could not link program: " + info);
            }
        }
        Log.d(TAG,"createProgram exit");
        return program;
    }

    public static void checkGlError(String op) {
        int error = GLES20.glGetError();
        if (error != GLES20.GL_NO_ERROR) {
            throw new RuntimeException(op + ": glError " + error);
        }
    }


public Boolean initProgram(){
    Log.d(TAG,"initProgram entry");
        mProgramId = createProgram(MERGE_VERTEX_SHADER, MERGE_FRAGMENT_SHADER);
    if (mProgramId == 0) {
        throw new RuntimeException("failed creating program");
    }

    aPosition = GLES20.glGetAttribLocation(mProgramId, "aPosition");
    aTexCoord = GLES20.glGetAttribLocation(mProgramId, "aTexCoord");
    rubyTexture1Y = GLES20.glGetUniformLocation(mProgramId, "rubyTexture1Y");
    rubyTexture1U = GLES20.glGetUniformLocation(mProgramId, "rubyTexture1U");
    rubyTexture1V = GLES20.glGetUniformLocation(mProgramId, "rubyTexture1V");
    rubyTexture2Y = GLES20.glGetUniformLocation(mProgramId, "rubyTexture2Y");
    rubyTexture2U = GLES20.glGetUniformLocation(mProgramId, "rubyTexture2U");
    rubyTexture2V = GLES20.glGetUniformLocation(mProgramId, "rubyTexture2V");
    rubyTexture3Y = GLES20.glGetUniformLocation(mProgramId, "rubyTexture3Y");
    rubyTexture3U = GLES20.glGetUniformLocation(mProgramId, "rubyTexture3U");
    rubyTexture3V = GLES20.glGetUniformLocation(mProgramId, "rubyTexture3V");
    rubyTexture4Y = GLES20.glGetUniformLocation(mProgramId, "rubyTexture4Y");
    rubyTexture4U = GLES20.glGetUniformLocation(mProgramId, "rubyTexture4U");
    rubyTexture4V = GLES20.glGetUniformLocation(mProgramId, "rubyTexture4V");
    rubyTexture5 = GLES20.glGetUniformLocation(mProgramId, "rubyTexture5");
    rubyTextureSize = GLES20.glGetUniformLocation(mProgramId, "rubyTextureSize");
    Log.d(TAG,"initProgram exit");
    return true;
}
    public void GLInit(){
        Log.d(TAG,"GLInit entry");
        int[] textures1 = new int[1];  //create no.of based on our input streams
        GLES20.glGenTextures(1, textures1,0);
        texture_map1 = textures1[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map1);

//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, texture_map, 0);

//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures2 = new int[1];
        GLES20.glGenTextures(1, textures2,0);
        texture_map2 = textures2[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map2);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        int[] textures3 = new int[1];
        GLES20.glGenTextures(1, textures3,0);
        texture_map3 = textures3[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map3);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures4 = new int[1];
        GLES20.glGenTextures(1, textures4,0);
        texture_map4 = textures4[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE3);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map4);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures5 = new int[1];
        GLES20.glGenTextures(1, textures5,0);
        texture_map5 = textures5[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map5);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures6 = new int[1];
        GLES20.glGenTextures(1, textures6,0);
        texture_map6 = textures6[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE5);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map6);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures7 = new int[1];
        GLES20.glGenTextures(1, textures7,0);
        texture_map7 = textures7[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE6);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map7);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures8 = new int[1];
        GLES20.glGenTextures(1, textures8,0);
        texture_map8 = textures8[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE7);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map8);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures9 = new int[1];
        GLES20.glGenTextures(1, textures9,0);
        texture_map9 = textures9[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE8);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map9);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures10 = new int[1];
        GLES20.glGenTextures(1, textures10,0);
        texture_map10 = textures10[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE9);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map9);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures11 = new int[1];
        GLES20.glGenTextures(1, textures11,0);
        texture_map11 = textures11[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE10);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map11);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures12 = new int[1];
        GLES20.glGenTextures(1, textures12,0);
        texture_map12 = textures12[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE11);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map12);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        int[] textures13 = new int[1];
        GLES20.glGenTextures(1, textures13,0);
        texture_map13 = textures13[0];
        GLES20.glActiveTexture(GLES20.GL_TEXTURE12);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map13);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);


        /*this is to read local file start*/
//        std::string FileName = std::string("/storage/emulated/0/opencvTesting/videoFrmImouInrawrgb24short.rgb");
//        fp = fopen(FileName.c_str(),"r+b");
//        if(NULL == fp) {
//            DPRINTF(" fopen() Error!!!\n");
//        }
//
//
//        freadbw = 1920;
//        freadbh = 1080 ;
//        fread_buf_size = freadbw*freadbh*3;
//        //Allocate Buffer for rawData
//        rawData = (unsigned char *)malloc(fread_buf_size);
//        if (NULL == rawData) {
//            DPRINTF("Rawdata is NULL\n");
//        }
        /*this is to read local file end*/

        mat_RGBA2YUV_I420 = new Mat() ;
        testConvertedImgYUVmat = new Mat();
        Log.d(TAG,"GLInit exit");
    }
    public void GLLoadShader(){
        initProgram();
    }

    public void ResizeOnSurfaceChange(int width, int height){
        // Set viewport
        //we would like to use input image size for screen
        Log.d(TAG,"ResizeOnSurfaceChange entry");
        GLES20.glViewport(0, 0, mScreenWidth, mScreenHeight);
        Log.d(TAG,"ResizeOnSurfaceChange exit");
    }

    public void GLDrawFrame() {
        Log.d(TAG, "GLDrawFrame entry");

        int[] max = new int[1];
        glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, max, 0);
        Log.d(TAG,"maxtextureimageunits:" + max[0]);

        int[] maxTextureSize = new int[1];
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, maxTextureSize, 0);
        Log.d(TAG,"maxTextureSize:" + maxTextureSize[0]);

        int[] maxrenderbufSize = new int[1];
        glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, maxrenderbufSize, 0);
        Log.d(TAG,"maxrenderbufSize:" + maxrenderbufSize[0]);

        int[] glbufSize = new int[1];
        glGetIntegerv(GL_BUFFER_SIZE, glbufSize, 0);
        Log.d(TAG,"glbufSize:" + glbufSize[0]);

        int[] glbufusage = new int[1];
        glGetIntegerv(GL_BUFFER_USAGE, glbufusage, 0);
        Log.d(TAG,"glbufusage:" + glbufusage[0]);




        float grey;
        grey = 0.00f;
//        GLES20.glClearColor(grey, grey, grey, 1.0f);
//        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
        Log.d(TAG, "GLDrawFrame glClear");
        int bw = 1920; //trying with hard code, later need to change
        int bh = 1080;

        Imgcodecs imageCodecs = new Imgcodecs();
        /**image one**/
        testInputImgJPEGmat =  imageCodecs.imread("/storage/emulated/0/opencvTesting/four_balls_color_jpg1920x1080.jpg",Imgcodecs.IMREAD_COLOR);;
        Log.d(TAG,"testInputImgJPEGmat channels:"+testInputImgJPEGmat.channels());
        Log.d(TAG,"testInputImgJPEGmat total:"+testInputImgJPEGmat.total());
        Log.d(TAG,"testInputImgJPEGmat elementSize:"+testInputImgJPEGmat.elemSize());
        Log.d(TAG,"testInputImgJPEGmat size:"+testInputImgJPEGmat.size());


        Imgproc.cvtColor(testInputImgJPEGmat, testConvertedImgYUVmat, Imgproc.COLOR_RGB2YUV_I420);
        Log.d(TAG,"testConvertedImgYUVmat channels:"+testConvertedImgYUVmat.channels());
        Log.d(TAG,"testConvertedImgYUVmat total:"+testConvertedImgYUVmat.total());
        Log.d(TAG,"testConvertedImgYUVmat elementSize:"+testConvertedImgYUVmat.elemSize());
        Log.d(TAG,"testConvertedImgYUVmat size:"+testConvertedImgYUVmat.size());

//        Imgcodecs.imwrite("/storage/emulated/0/opencvTesting/mygltest/testConvertedImgYUVmat.jpg", testConvertedImgYUVmat);

        Log.d(TAG, "GLDrawFrame after bitmap read local jpeg");
//        int LENGTH = testConvertedImgYUVmat.rows()*testConvertedImgYUVmat.cols();
        int LENGTH = 1920*1080;
        byte [] ydataarray = new byte[LENGTH];
        byte [] Udataarray = new byte[LENGTH/4];
        byte [] Vdataarray = new byte[LENGTH/4];
        int U_INDEX = LENGTH;
        int V_INDEX = LENGTH*5/4;
        int totalbytesinmat = (int)(testConvertedImgYUVmat.total()*testConvertedImgYUVmat.channels());
        byte [] totaldatafrommat = new byte[totalbytesinmat];
        testConvertedImgYUVmat.get(0,0,totaldatafrommat);
        System.arraycopy(totaldatafrommat, 0, ydataarray, 0, LENGTH);

        System.arraycopy(totaldatafrommat, U_INDEX, Udataarray, 0, LENGTH/4);
        System.arraycopy(totaldatafrommat, V_INDEX, Vdataarray, 0, LENGTH/4);
        ByteBuffer yByteBuffer = ByteBuffer.allocateDirect(LENGTH);
        ByteBuffer uByteBuffer = ByteBuffer.allocateDirect(LENGTH/4);
        ByteBuffer vByteBuffer = ByteBuffer.allocateDirect(LENGTH/4);
        yByteBuffer.put(ydataarray);
        yByteBuffer.position(0);
        uByteBuffer.put(Udataarray);
        uByteBuffer.position(0);
        vByteBuffer.put(Vdataarray);
        vByteBuffer.position(0);
/**image one end**/

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map1);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
//        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, imgbitmap, 0);  //working with bitmap
        Log.d(TAG, "GLDrawFrame after first texture0");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map2);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, uByteBuffer);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
        Log.d(TAG, "GLDrawFrame after second texture1");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map3);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, vByteBuffer);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
        Log.d(TAG, "GLDrawFrame after third texture2");

        /**second image start **/
        testInputImgJPEGmat = imageCodecs.imread("/storage/emulated/0/opencvTesting/apple-jpg-1920x1080.jpg",Imgcodecs.IMREAD_COLOR);
        Imgproc.cvtColor(testInputImgJPEGmat, testConvertedImgYUVmat, Imgproc.COLOR_RGB2YUV_I420);
        testConvertedImgYUVmat.get(0,0,totaldatafrommat);
        System.arraycopy(totaldatafrommat, 0, ydataarray, 0, LENGTH);
        System.arraycopy(totaldatafrommat, U_INDEX, Udataarray, 0, LENGTH/4);
        System.arraycopy(totaldatafrommat, V_INDEX, Vdataarray, 0, LENGTH/4);
        yByteBuffer.put(ydataarray);
        yByteBuffer.position(0);
        uByteBuffer.put(Udataarray);
        uByteBuffer.position(0);
        vByteBuffer.put(Vdataarray);
        vByteBuffer.position(0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE3);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map4);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE4);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map5);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, uByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE5);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map6);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, vByteBuffer);

        /**3rd image start **/
        testInputImgJPEGmat = imageCodecs.imread("/storage/emulated/0/opencvTesting/lappy-jpg-1920x1080.jpg",Imgcodecs.IMREAD_COLOR);
        Imgproc.cvtColor(testInputImgJPEGmat, testConvertedImgYUVmat, Imgproc.COLOR_RGB2YUV_I420);
        testConvertedImgYUVmat.get(0,0,totaldatafrommat);
        System.arraycopy(totaldatafrommat, 0, ydataarray, 0, LENGTH);
        System.arraycopy(totaldatafrommat, U_INDEX, Udataarray, 0, LENGTH/4);
        System.arraycopy(totaldatafrommat, V_INDEX, Vdataarray, 0, LENGTH/4);
        yByteBuffer.put(ydataarray);
        yByteBuffer.position(0);
        uByteBuffer.put(Udataarray);
        uByteBuffer.position(0);
        vByteBuffer.put(Vdataarray);
        vByteBuffer.position(0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE6);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map7);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE7);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map8);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, uByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE8);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map9);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, vByteBuffer);

        /**4th image start **/
        testInputImgJPEGmat = imageCodecs.imread("/storage/emulated/0/opencvTesting/wpinjpg1920x1080.jpeg",Imgcodecs.IMREAD_COLOR);
        Imgproc.cvtColor(testInputImgJPEGmat, testConvertedImgYUVmat, Imgproc.COLOR_RGB2YUV_I420);
        testConvertedImgYUVmat.get(0,0,totaldatafrommat);
        System.arraycopy(totaldatafrommat, 0, ydataarray, 0, LENGTH);
        System.arraycopy(totaldatafrommat, U_INDEX, Udataarray, 0, LENGTH/4);
        System.arraycopy(totaldatafrommat, V_INDEX, Vdataarray, 0, LENGTH/4);
        yByteBuffer.put(ydataarray);
        yByteBuffer.position(0);
        uByteBuffer.put(Udataarray);
        uByteBuffer.position(0);
        vByteBuffer.put(Vdataarray);
        vByteBuffer.position(0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE9);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map10);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw, bh, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE10);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map11);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, uByteBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE11);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map12);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, bw/2, bh/2, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, vByteBuffer);

        /**5th image in rgb form start **/
        testInputImgJPEGmat = imageCodecs.imread("/storage/emulated/0/opencvTesting/wpinjpg1920x1080.jpeg",Imgcodecs.IMREAD_COLOR);
//        Imgproc.cvtColor(testInputImgJPEGmat, testConvertedImgYUVmat, Imgproc.COLOR_RGB2YUV_I420);
        int totalbytesinrgbmat = (int)(testInputImgJPEGmat.total()*testInputImgJPEGmat.channels());
        byte [] totaldatafromrgbmat = new byte[totalbytesinrgbmat];
        testInputImgJPEGmat.get(0,0,totaldatafromrgbmat);
        byte [] rgbdataarray = new byte[totalbytesinrgbmat];
        System.arraycopy(totaldatafromrgbmat, 0, rgbdataarray, 0, totalbytesinrgbmat);
        ByteBuffer rgbByteBuffer = ByteBuffer.allocateDirect(totalbytesinrgbmat);
        rgbByteBuffer.put(rgbdataarray);
        rgbByteBuffer.position(0);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE12);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map13);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, bw, bh, 0, GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, rgbByteBuffer);

        GLES20.glUseProgram(mProgramId);
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, mPosTriangleVertices);
        GLES20.glEnableVertexAttribArray(aPosition);

        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 0, mTexVertices);
        GLES20.glEnableVertexAttribArray(aTexCoord);

        GLES20.glUniform2f(rubyTextureSize, bw, bh);
        GLES20.glUniform1i(rubyTexture1Y, 0);
        GLES20.glUniform1i(rubyTexture1U, 1);
        GLES20.glUniform1i(rubyTexture1V, 2);
        GLES20.glUniform1i(rubyTexture2Y, 3);
        GLES20.glUniform1i(rubyTexture2U, 4);
        GLES20.glUniform1i(rubyTexture2V, 5);
        GLES20.glUniform1i(rubyTexture3Y, 6);
        GLES20.glUniform1i(rubyTexture3U, 7);
        GLES20.glUniform1i(rubyTexture3V, 8);
        GLES20.glUniform1i(rubyTexture4Y, 9);
        GLES20.glUniform1i(rubyTexture4U, 10);
        GLES20.glUniform1i(rubyTexture4V, 11);
        GLES20.glUniform1i(rubyTexture5, 12);


//    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  //for single input
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 24);   // for four inputs

        ByteBuffer pReadPixelBuf;
        pReadPixelBuf = ByteBuffer.allocateDirect(mScreenHeight * mScreenWidth * 4);
        pReadPixelBuf.order(ByteOrder.LITTLE_ENDIAN);

        GLES20.glReadPixels(0, 0, mScreenWidth, mScreenHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pReadPixelBuf);

            outputReadpixelInMat = new Mat(mScreenHeight,mScreenWidth, CvType.CV_8UC4,pReadPixelBuf);
        Imgproc.cvtColor(outputReadpixelInMat, mat_RGBA2YUV_I420, Imgproc.COLOR_RGBA2YUV_I420);

            Imgcodecs.imwrite("/storage/emulated/0/opencvTesting/mygltest/Javmat_RGBA2YUV_I420.jpg", mat_RGBA2YUV_I420);

        Bitmap bitmapoutputFrmReadPixel = Bitmap.createBitmap(mScreenWidth, mScreenHeight, Bitmap.Config.ARGB_8888);
        bitmapoutputFrmReadPixel.copyPixelsFromBuffer(pReadPixelBuf);

//            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
//            bitmapoutputFrmReadPixel.compress(Bitmap.CompressFormat.PNG, 40, bytes);

//you can create a new file name "test.BMP" in sdcard folder.
//            File filepath = new File(Environment.getExternalStorageDirectory();
            File  dirpath=new File("/storage/emulated/0/opencvTesting/mygltest/");
            dirpath.mkdirs();
            File glreadfile=new File(dirpath,"myglreadpixel.jpg");

            OutputStream out=null;
            try{
                out=new FileOutputStream(glreadfile);
                bitmapoutputFrmReadPixel.compress(Bitmap.CompressFormat.JPEG,100,out);
                out.flush();
                out.close();


//                MediaStore.Images.Media.insertImage(getContentResolver(), bitmapoutputFrmReadPixel," yourTitle "," yourDescription");

                bitmapoutputFrmReadPixel=null;


            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        Log.d(TAG, "GLDrawFrame exit");

    }
}
