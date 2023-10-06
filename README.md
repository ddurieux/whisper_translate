# Real Time Whisper Translation

## Adaptation

This script is an adapation of the script from https://github.com/davabase/whisper_real_time

## Introduction

This script transcript english audio and translate in french, need chatGPT account from the translation part.

It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.


## Requirements

The requirements are:

* an NVIDIA card with CUDA cores (like RTX...)
* python 3
* ffmpeg

## Installation

You need install:

* install [ffmpeg](https://ffmpeg.org/), on windows, add the bin path to the environment *PATH*
* install CUDA toolkit https://developer.nvidia.com/cuda-toolkit, **YOU NEED INSTALL IT BEFORE PYTHON LIBS DEFINED AT NEXT BULLET**
* install [git](https://git-scm.com/downloads)
* Install python dependencies:
    ```
    pip install -r requirements.txt
    ```
* For chatGPT translation edit the file *transcribe.py* and set the key in the begining of the file.


## Usage

List audio devices with command:

```
python transcribe.py --default_microphone list
```

Choose the device you want, for exemple "mic" and run:

```
python transcribe.py --default_microphone "mic" --model small
```

If you graphic card has 10 Go of memory and more, you can replace *small* by *medium*

## Integrate translation into OBS

The script write on screen, but it write too in file *subtitle.txt* in current folder.

IN OBS, add *text GDI+*, check *Chatlog Mode* and the line limit to 3.


## Special case: translate streaming video

### Linux

Under linux, you can select the sound card device named *pipewire*.

### Windows

* install [VB-Audio](https://vb-audio.com/Cable/) in case you want translate streaming video and not only your microphone
* manage in "app volume and device preferences" in windows settings to put your browser video into *Cable Input (VB-Audio Virtual Cable)*
