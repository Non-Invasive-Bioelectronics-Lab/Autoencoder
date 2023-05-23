package com.example.autoencoderapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.resources.Compatibility;

import android.content.Context;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


import com.example.autoencoderapp.ml.AutoencodercnnFinal;
import com.example.autoencoderapp.ml.AutoencodercnnV4;
import com.giftedcat.wavelib.view.WaveView;
//import com.google.android.gms.tflite.gpu.GpuDelegate;

import org.apache.commons.collections.ListUtils;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
//import org.tensorflow.lite.Delegate;
//import org.tensorflow.lite.Interpreter;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
//import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity {


    // Data
    List<Float> rawDataArray = new ArrayList<Float>();  // read data from .csv, and convert string to double ArrayList
    float[] eegData;
    float[] data;
    float[] eegSegment;
    List<Float> outputListAll = new ArrayList<Float>();


    // Model
//    AutoencoderModel autoencoderModel;


    // waveView 1: raw EEG data;  waveView2: denoised EEG data
    // waveUtil_editable1: raw EEG data, waveUtil_editable2: denoised EEG data
    WaveView waveView1, waveView2;
    WaveUtil_Editable waveUtil_editable1, waveUtil_editable2;
    TextView textView1, textView2, textViewX1, textViewX2, textViewXName1, textViewXName2;
    Button testButton, runOnCPUButton, runOnGPUButton;
    Boolean isTestRunning = false;
    int totalLoop;


    // set up 'Timer' for real-time signal processing
//    private Timer timer;
//    private TimerTask timerTask;
//    int loopCount=0;


    private Timer timer = new Timer();
//    AutoencodercnnV4 model;               // This an early version of autoencoder model, it works nut worse than the final model
    AutoencodercnnFinal model;              // This is the final autoencoder model, it works well, and it's in the journal publication

    private String filename;
//    private String filename = "Autoencoder_CPU.txt";

    Model.Options options;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ///////////////////// Set up Views //////////////////////
        // Assign all widgets
        waveView1 = findViewById(R.id.waveView1);
        waveView2 = findViewById(R.id.waveView2);
        textView1 = findViewById(R.id.textView1);
        textView2 = findViewById(R.id.textView2);
        testButton = findViewById(R.id.testButton);
        runOnCPUButton = findViewById(R.id.runOnCPUButton);
        runOnGPUButton = findViewById(R.id.runOnGPUButton);

        textViewX1 = findViewById(R.id.textViewX1);
        textViewX2 = findViewById(R.id.textViewX2);
        textViewXName1 = findViewById(R.id.textViewXName1);
        textViewXName2 = findViewById(R.id.textViewXName2);



        // Create waveUtil_editable objects
        waveUtil_editable1 = new WaveUtil_Editable();
        waveUtil_editable2 = new WaveUtil_Editable();

        // adjust time domain view, smaller value leads to more compact view
        waveView1.setWaveLineWidth(5);
        waveView2.setWaveLineWidth(5);


        ///////////////////// Data processing /////////////////////
        // read pre-stored raw EEG data
        readDataInternalStroage();

//        for (int i=0; i<50; i++){
//            System.out.println("data = " + rawDataArray.get(i));
//        }
//        System.out.println("ARRAYLIST length = "+ rawDataArray.size()/200 +"seconds");

        // Convert List to array
        eegData = new float[rawDataArray.size()];
        for (int i=0; i< eegData.length; i++){
            eegData[i]=rawDataArray.get(i);
        }


        totalLoop = (int)eegData.length/800;
        System.out.println("EEG data length is = " + eegData.length);
        System.out.println("Total Loop is = " + totalLoop);



        //////************** Just run and plot ******************////////
        for (int i=0; i<totalLoop; i++){
            System.out.println("Loop Count: " + i);
            eegSegment = data_segment_normalize(eegData, i);

            float[] output = new float[800];

            List<Float> outputList = new ArrayList<Float>(800);

            try {
                // calculate processing time
                model = AutoencodercnnFinal.newInstance(MainActivity.this);
                timer.StartTimer();

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);

                // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800*4);
                byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
                for(int j=0; j< eegSegment.length;j++){
                    byteBuffer.putFloat(eegSegment[j]);
                }

                inputFeature0.loadBuffer(byteBuffer);

                // Runs model inference and gets result.
                AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                output = outputFeature0.getFloatArray();

                for (int k=0; k< eegSegment.length; k++){
                    outputList.add(output[k]);
                    //System.out.println("Output is: "+output[i]);

                    outputListAll.add(output[k]);
                }

                // Releases model resources if no longer used.
                model.close();

                System.out.println("TFLite Autoencoder processing time for each segment: "+ timer.ElapsedTime() + "s");
                saveTxt(String.valueOf(timer.ElapsedTime()));

                System.out.println("Loop is finished! and the outputList Length = " + outputList.size());

            } catch (IOException e) {
                // TODO Handle the exception
                e.printStackTrace();
            }

            System.out.println("outputListAll length is: " + outputListAll.size());

        }



        ////// GPU Acceleration
        //// Initialize interpreter with GPU delegate
//        CompatibilityList compatList = new CompatibilityList();
//
//        if(compatList.isDelegateSupportedOnThisDevice()){
//            // if the device has a supported GPU, add the GPU delegate
//            options = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
//        } else {
//            // if the GPU is not supported, run on 4 threads
//            options = new Model.Options.Builder().setNumThreads(4).build();
//        }
//
//
//        filename = "gpubattery111.txt";
//        java.util.Timer timer1 = new java.util.Timer();
//        TimerTask timerTask = new TimerTask() {
//            @Override
//            public void run() {
//                for (int i=0; i<totalLoop; i++){
//                    System.out.println("Loop Count: " + i);
//                    eegSegment = data_segment_normalize(eegData, i);
//
//                    float[] output = new float[800];
//
//                    List<Float> outputList = new ArrayList<Float>(800);
//
//                    try {
//                        // calculate processing time
//                        model = AutoencodercnnFinal.newInstance(MainActivity.this, options);
//                        timer.StartTimer();
//
//                        // Creates inputs for reference.
//                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);
//
//                        // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
//                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800*4);
//                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
//                        for(int j=0; j< eegSegment.length;j++){
//                            byteBuffer.putFloat(eegSegment[j]);
//                        }
//
//                        inputFeature0.loadBuffer(byteBuffer);
//
//                        // Runs model inference and gets result.
//                        AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
//                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                        output = outputFeature0.getFloatArray();
//
//                        for (int k=0; k< eegSegment.length; k++){
//                            outputList.add(output[k]);
//                            //System.out.println("Output is: "+output[i]);
//
//                            outputListAll.add(output[k]);
//                        }
//
//                        // Releases model resources if no longer used.
//                        model.close();
//
//                        System.out.println("TFLite Autoencoder processing time for each segment: "+ timer.ElapsedTime() + "s");
//                        saveTxt(String.valueOf(timer.ElapsedTime()));
//
//                        System.out.println("Loop is finished! and the outputList Length = " + outputList.size());
//
//                    } catch (IOException e) {
//                        // TODO Handle the exception
//                        e.printStackTrace();
//                    }
//
//                    System.out.println("outputListAll length is: " + outputListAll.size());
//
//                }
//
//            }
//        };
//        timer1.schedule(timerTask, 1000, 1000*5);








        ////Main code for running tensorflow lite model --- CPU Running
//        filename = "cpuBattery111.txt";
//        java.util.Timer timer1 = new java.util.Timer();
//        TimerTask timerTask = new TimerTask() {
//            @Override
//            public void run() {
//                for (int i=0; i<totalLoop; i++){
//                    System.out.println("Loop Count: " + i);
//                    eegSegment = data_segment_normalize(eegData, i);
//
//                    float[] output = new float[800];
//
//                    List<Float> outputList = new ArrayList<Float>(800);
//
//                    try {
//                        // calculate processing time
//                        model = AutoencodercnnFinal.newInstance(MainActivity.this);
//                        timer.StartTimer();
//
//                        // Creates inputs for reference.
//                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);
//
//                        // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
//                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800*4);
//                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
//                        for(int j=0; j< eegSegment.length;j++){
//                            byteBuffer.putFloat(eegSegment[j]);
//                        }
//
//                        inputFeature0.loadBuffer(byteBuffer);
//
//                        // Runs model inference and gets result.
//                        AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
//                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                        output = outputFeature0.getFloatArray();
//
//                        for (int k=0; k< eegSegment.length; k++){
//                            outputList.add(output[k]);
//                            //System.out.println("Output is: "+output[i]);
//
//                            outputListAll.add(output[k]);
//                        }
//
//                        // Releases model resources if no longer used.
//                        model.close();
//
//                        System.out.println("TFLite Autoencoder processing time for each segment: "+ timer.ElapsedTime() + "s");
//                        saveTxt(String.valueOf(timer.ElapsedTime()));
//
//                        System.out.println("Loop is finished! and the outputList Length = " + outputList.size());
//
//                    } catch (IOException e) {
//                        // TODO Handle the exception
//                        e.printStackTrace();
//                    }
//
//                    System.out.println("outputListAll length is: " + outputListAll.size());
//
//                }
//            }
//        };
//
//        timer1.schedule(timerTask, 1000, 1000*5);









        testButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isTestRunning){
                    waveUtil_editable1.showWaveData(waveView1, rawDataArray);
                    waveUtil_editable2.showWaveData(waveView2, outputListAll);
                    testButton.setText("Pause");
                }else{
                    waveUtil_editable1.stop();
                    waveUtil_editable2.stop();
                    testButton.setText("Start Test");
                }
                isTestRunning=!isTestRunning;
            }
        });






        //******************* Porcessing time VS Signal duration **********************
//        filename = "CPU2400.txt";
//        int duration = 2400;
//        java.util.Timer timer1 = new java.util.Timer();
//        TimerTask timerTask = new TimerTask() {
//            @Override
//            public void run() {
//                // initialize a single channel EEG signal
//                float simulatedEEG[] = new float[duration];
//                // simulate the signal by Random
//                Random random = new Random();
//                for (int i=0; i<duration; i++){
//                    simulatedEEG[i] = (float) (100*(random.nextFloat()-0.5));
//                }
//                System.out.println("simulatedEEG length: " +simulatedEEG.length);
//
//                int loopCount= (int) simulatedEEG.length/800;
//                System.out.println("Total Loop : " + loopCount);
//
//
////                double totalTime=0;
//
//                for(int i=0; i<loopCount; i++){
//                    eegSegment = data_segment_normalize(simulatedEEG, i);
//
//                    float[] output = new float[800];
//                    List<Float> outputList = new ArrayList<Float>(800);
//
//                    try {
//                        // calculate processing time
//                        model = AutoencodercnnFinal.newInstance(MainActivity.this);
//                        timer.StartTimer();
//
//                        // Creates inputs for reference.
//                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);
//
//                        // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
//                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800*4);
//                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
//                        for(int j=0; j<eegSegment.length;j++){
//                            byteBuffer.putFloat(eegSegment[j]);
//                        }
//
//                        inputFeature0.loadBuffer(byteBuffer);
//
//                        // Runs model inference and gets result.
//                        AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
//                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                        output = outputFeature0.getFloatArray();
//
//                        for (int k=0; k< eegSegment.length; k++){
//                            outputList.add(output[k]);
//                            outputListAll.add(output[k]);
//                        }
//
//                        // Releases model resources if no longer used.
//                        model.close();
//
//                        System.out.println("TFLite Autoencoder processing time: "+ timer.ElapsedTime() + "s");
//                        saveTxt(String.valueOf(timer.ElapsedTime()));
//
////                        totalTime = totalTime+timer.ElapsedTime();
//
//                    } catch (IOException e) {
//                        // TODO Handle the exception
//                        e.printStackTrace();
//                    }
//
//                }
//
////                System.out.println("TFLite total processing time: "+ totalTime + "s");
//
//
//            }
//        };
//        timer1.schedule(timerTask, 1000, 1000);





    }// This is the onCreate method bracket






    public void setRunOnCPUButton(View view) {
        filename = "Autoencoder_CPU.txt";

        java.util.Timer timer1 = new java.util.Timer();
        TimerTask timerTask1 = new TimerTask() {
            @Override
            public void run() {
                for (int i = 0; i < totalLoop; i++) {
                    System.out.println("Loop Count: " + i);
                    eegSegment = data_segment_normalize(eegData, i);

                    float[] output = new float[800];

                    List<Float> outputList = new ArrayList<Float>(800);

                    try {
                        // calculate processing time
                        model = AutoencodercnnFinal.newInstance(MainActivity.this);
                        timer.StartTimer();

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);

                        // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800 * 4);
                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
                        for (int j = 0; j < eegSegment.length; j++) {
                            byteBuffer.putFloat(eegSegment[j]);
                        }

                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        output = outputFeature0.getFloatArray();

                        for (int k = 0; k < eegSegment.length; k++) {
                            outputList.add(output[k]);
                            //System.out.println("Output is: "+output[i]);

                            outputListAll.add(output[k]);
                        }

                        // Releases model resources if no longer used.
                        model.close();

                        System.out.println("TFLite Autoencoder processing time for each segment: " + timer.ElapsedTime() + "s");
                        saveTxt(String.valueOf(timer.ElapsedTime()));

                        System.out.println("Loop is finished! and the outputList Length = " + outputList.size());

                    } catch (IOException e) {
                        // TODO Handle the exception
                        e.printStackTrace();
                    }

                    System.out.println("outputListAll length is: " + outputListAll.size());
                }
            }
        };

        timer1.schedule(timerTask1, 1000, 1000*5);

    }



    public void setRunOnGPUButton(View view) {
        filename = "Autoencoder_GPU.txt";

        java.util.Timer timer1 = new java.util.Timer();
        TimerTask timerTask1 = new TimerTask() {
            @Override
            public void run() {
                for (int i = 0; i < totalLoop; i++) {
                    System.out.println("Loop Count: " + i);
                    eegSegment = data_segment_normalize(eegData, i);

                    float[] output = new float[800];

                    List<Float> outputList = new ArrayList<Float>(800);

                    try {
                        // calculate processing time
                        model = AutoencodercnnFinal.newInstance(MainActivity.this, options);
                        timer.StartTimer();

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 800}, DataType.FLOAT32);

                        // Here, need to store input tensor into 'ByteBuffer', little-endian (tflite works only in this format)
                        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(800 * 4);
                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
                        for (int j = 0; j < eegSegment.length; j++) {
                            byteBuffer.putFloat(eegSegment[j]);
                        }

                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        AutoencodercnnFinal.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        output = outputFeature0.getFloatArray();

                        for (int k = 0; k < eegSegment.length; k++) {
                            outputList.add(output[k]);
                            //System.out.println("Output is: "+output[i]);

                            outputListAll.add(output[k]);
                        }

                        // Releases model resources if no longer used.
                        model.close();

                        System.out.println("TFLite Autoencoder processing time for each segment: " + timer.ElapsedTime() + "s");
                        saveTxt(String.valueOf(timer.ElapsedTime()));

                        System.out.println("Loop is finished! and the outputList Length = " + outputList.size());

                    } catch (IOException e) {
                        // TODO Handle the exception
                        e.printStackTrace();
                    }

                    System.out.println("outputListAll length is: " + outputListAll.size());
                }
            }
        };

        timer1.schedule(timerTask1, 1000, 1000*5);

    }



    private float[] data_segment_normalize(float[] data, int loopCount){

        // the 'data' param here is the whole EEG recording, having 53-second
        int segmentStartPoint=loopCount*800;
        int segment_length=800;
        float[] eegSegment = Arrays.copyOfRange(data, segmentStartPoint, segmentStartPoint+segment_length);
        float[] segment_sorted = Arrays.copyOfRange(data, segmentStartPoint, segmentStartPoint+segment_length);

        Arrays.sort(segment_sorted);
        float maxValue = segment_sorted[segment_sorted.length-1];
        float minValue = segment_sorted[0];
//        System.out.println("max = " +maxValue + "min = " +minValue);

        //Normalization
        for (int i=0; i<segment_length; i++){
            eegSegment[i] = (eegSegment[i]-minValue)/(maxValue-minValue);
//             System.out.println("Data segment normalized: " + segment[i]);
        }

        return eegSegment;
    }


    private float[] data_segment(float[] data){
        int segment_length=800;
        float[] segment = Arrays.copyOfRange(eegData, 0, segment_length);
        float[] segment_sorted = Arrays.copyOfRange(eegData, 0, segment_length);

        Arrays.sort(segment_sorted);
        float maxValue = segment_sorted[segment_sorted.length-1];
        float minValue = segment_sorted[0];
//        System.out.println("max = " +maxValue + "min = " +minValue);

        //Normalization
        for (int i=0; i<segment_length; i++){
             segment[i] = (segment[i]-minValue)/(maxValue-minValue);
//             System.out.println("Data segment normalized: " + segment[i]);
        }
        return segment;
    }


    public void readDataInternalStroage(){

        BufferedInputStream bufferedInputStream = new BufferedInputStream(getResources().openRawResource(R.raw.syntheticxx));
        BufferedReader bufferedReader = new BufferedReader(
                new InputStreamReader(bufferedInputStream));

        String rawData = null;

        try {
            while ((rawData = bufferedReader.readLine()) != null) {
//                rawDataArray.add( Double.parseDouble(rawData));
                rawDataArray.add(Float.parseFloat(rawData));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



//    public void setTestButton(View view) {
//        if (!isTestRunning){
//            waveUtil_editable1.showWaveData(waveView1, rawDataArray);
////            waveUtil_editable2.showWaveDatas(10, waveView2, rawDataArray);
//            waveUtil_editable2.showWaveData(waveView2, rawDataArray);
//            testButton.setText("Pause");
//        }else{
//            waveUtil_editable1.stop();
//            waveUtil_editable2.stop();
//            testButton.setText("Start Test");
//        }
//        isTestRunning=!isTestRunning;
//    }




    public void saveTxt(String usageLog) {

        String filepath = (Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath());

        // create a new filename
        File file = new File(filepath + File.separator + filename);
        try{

            FileOutputStream fos = new FileOutputStream(file, true);
            fos.write(System.getProperty("line.separator").getBytes());
            fos.write(usageLog.getBytes(StandardCharsets.UTF_8));


        } catch(IOException e) {
            e.printStackTrace();
                Toast.makeText(this,"Error! File not saved", Toast.LENGTH_SHORT).show();
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        waveUtil_editable1.stop();
        waveUtil_editable2.stop();
    }







}