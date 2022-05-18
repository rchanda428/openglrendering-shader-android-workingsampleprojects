package com.android.gl2jni;
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
    private int mScreenWidth  = 1920;
    private int mScreenHeight = 1080;
    private int mProgramId;
    private int aPosition;
    private int aTexCoord;
    private int rubyTexture1;
    private int rubyTexture2;
    private int rubyTexture3;
    private int rubyTexture4;

    private int rubyTextureSize;
    private int texture_map1;
    private int texture_map2;
    private int texture_map3;
    private int texture_map4;
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
                    "void main() {\n" +
                    "  vTexCoord = aTexCoord;\n" +
                    "  gl_Position = vec4(aPosition.x,-aPosition.y, 0.0, 1.0);\n" +
                    "}\n";

    private static final String MERGE_FRAGMENT_SHADER =
                    "#ifdef GL_FRAGMENT_PRECISION_HIGH\n" +
                    "precision highp float;\n" +
                    "#else\n" +
                    "precision mediump float;\n" +
                    "#endif\n" +
                    "uniform sampler2D rubyTexture1;\n" +
                    "uniform sampler2D rubyTexture2;\n" +
                    "uniform sampler2D rubyTexture3;\n" +
                    "uniform sampler2D rubyTexture4;\n" +
                    "varying vec2 vTexCoord;\n" +
                    "void main() {\n" +
                    "vec4  i = (texture2D(rubyTexture1, vTexCoord)+texture2D(rubyTexture2,vTexCoord)+texture2D(rubyTexture3,vTexCoord)+texture2D(rubyTexture4,vTexCoord));\n" +
                    "gl_FragColor.rgba = i;\n" +
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
    rubyTexture1 = GLES20.glGetUniformLocation(mProgramId, "rubyTexture1");
    rubyTexture2 = GLES20.glGetUniformLocation(mProgramId, "rubyTexture2");

    rubyTexture3 = GLES20.glGetUniformLocation(mProgramId, "rubyTexture3");
    rubyTexture4 = GLES20.glGetUniformLocation(mProgramId, "rubyTexture4");

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
        float grey;
        grey = 0.00f;
//        GLES20.glClearColor(grey, grey, grey, 1.0f);
//        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
        Log.d(TAG, "GLDrawFrame glClear");
        int bw = 1920; //trying with hard code, later need to change
        int bh = 1080;
        Bitmap imgbitmap;
//        ByteBuffer imgbitmapBuffer = ByteBuffer.allocateDirect(bw * bh * 4).order(ByteOrder.nativeOrder());//RGBA
        ByteBuffer imgbitmapBuffer = ByteBuffer.allocateDirect(bw * bh * 4).order(ByteOrder.nativeOrder());//RGBA
        byte[] originalMap = new byte[(bw * bh)];

        try {
            imgbitmap = BitmapFactory.decodeStream(mAssetContext.getAssets().open("four_balls_color_jpg1920x1080.jpg"));
            bw = imgbitmap.getWidth();
            bh = imgbitmap.getHeight();
            Log.d(TAG, "Bitmap size = " + imgbitmap.getByteCount());
            Log.i(TAG, "Buffer size = " + imgbitmapBuffer.capacity());
            Log.d(TAG, "bw:" + bw);
            Log.d(TAG, "bh:" + bh);
//            imgbitmap.copyPixelsToBuffer(imgbitmapBuffer);
//            imgbitmap.get(originalMap, 0, imgbitmap.capacity()); //not working
//            ByteBuffer originalMapBuffer = ByteBuffer.wrap(originalMap); //not working



        Log.d(TAG, "GLDrawFrame after bitmap read local jpeg");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map1);
//    glBindFramebuffer(GL_FRAMEBUFFER, iFrameBuffObject);
//    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, texture_map, 0);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, bw, bh, 0, GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, imgbitmapBuffer);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, imgbitmap, 0);  //working with bitmap
        Log.d(TAG, "GLDrawFrame after first texture0");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map2);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, bw, bh, 0, GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, imgbitmapBuffer);

        Log.d(TAG, "GLDrawFrame after second texture1");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map3);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, bw, bh, 0, GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, imgbitmapBuffer);

        Log.d(TAG, "GLDrawFrame after third texture2");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE3);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture_map4);
//        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGB, bw, bh, 0, GLES20.GL_RGB, GLES20.GL_UNSIGNED_BYTE, imgbitmapBuffer);

        Log.d(TAG, "GLDrawFrame after first texture3");

        GLES20.glUseProgram(mProgramId);

        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, mPosTriangleVertices);
        GLES20.glEnableVertexAttribArray(aPosition);

        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 0, mTexVertices);
        GLES20.glEnableVertexAttribArray(aTexCoord);

        GLES20.glUniform2f(rubyTextureSize, bw, bh);
        GLES20.glUniform1i(rubyTexture1, 0);
        GLES20.glUniform1i(rubyTexture2, 1);
        GLES20.glUniform1i(rubyTexture3, 2);
        GLES20.glUniform1i(rubyTexture4, 3);


//    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  //for single input
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 24);   // for four inputs

        ByteBuffer pReadPixelBuf;
        pReadPixelBuf = ByteBuffer.allocateDirect(mScreenHeight * mScreenWidth * 4);
        pReadPixelBuf.order(ByteOrder.LITTLE_ENDIAN);

        GLES20.glReadPixels(0, 0, mScreenWidth, mScreenHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pReadPixelBuf);

            outputReadpixelInMat = new Mat(mScreenHeight,mScreenWidth, CvType.CV_8UC4,pReadPixelBuf);
        Imgproc.cvtColor(outputReadpixelInMat, mat_RGBA2YUV_I420, Imgproc.COLOR_RGBA2YUV_I420);

            Imgcodecs.imwrite("/storage/emulated/0/opencvTesting/Javmat_RGBA2YUV_I420.jpg", mat_RGBA2YUV_I420);

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
         catch (IOException e) {
            e.printStackTrace();
            Log.d("Tag",e.getMessage());
        }
    }
}
