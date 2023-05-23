package com.example.autoencoderapp;

public class Timer {

    // Fields
    private double _tElapsed=0.0;
    private double _t0=0.0;

    // Methods

    public double ElapsedTime(){

        _tElapsed=System.currentTimeMillis()-_t0;       //ms
        _tElapsed=_tElapsed*Math.pow(10,-3);            //s

        return _tElapsed;
    }

    public void StartTimer(){
        //Reset time to 0s
        _t0=System.currentTimeMillis();                 //ms     Time at 0 i.e. when start function called (Timer Reset functionality)
    }


}
