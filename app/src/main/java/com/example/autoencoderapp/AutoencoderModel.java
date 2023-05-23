package com.example.autoencoderapp;

import android.content.Context;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class AutoencoderModel {




    //Load instance of interpreter from TFlite Android support library
    protected Interpreter interpreter;
    private static final String Model_Name = "autoencoderCNN.tflite"; //Load model for testing (float-32)
    public float[] output;
//    public int[] output_shape;
//    public int output_index = 0;




    //Load Model into Interpreter
    public AutoencoderModel(Context context) throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, Model_Name);
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        interpreter = new Interpreter(tfliteModel, tfliteOptions);
    }


    public float[] predictions(float[] input_data) {
        float[] results = new float[800]; //Output shape of results
        interpreter.run(input_data, results); //Run Interpreter on data input from Main Activity
        return results; //Return prediction results
    }





}
