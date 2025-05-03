package com.os.cvCamera

import android.content.Context
import android.media.MediaPlayer
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.CvType.CV_8UC1
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY
import org.opencv.imgproc.Imgproc.Canny
import org.opencv.imgproc.Imgproc.Sobel
import org.opencv.imgproc.Imgproc.cvtColor

fun Mat.toSobel(): Mat {
    Sobel(this, this, CV_8UC1, 1, 0)
    return this
}

fun Mat.toSepia(): Mat {
    val sepiaKernel = Mat(4, 4, CvType.CV_32F)
    sepiaKernel.put(
        0,
        0,
        0.189, 0.769, 0.393, 0.0,
        0.168, 0.686, 0.349, 0.0,
        0.131, 0.534, 0.272, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    Core.transform(this, this, sepiaKernel)
    return this
}

fun Mat.toGray(): Mat {
    cvtColor(this, this, COLOR_BGR2GRAY)
    return this
}

fun Mat.toCanny(): Mat {
    val tmpMat = Mat()
    Canny(this, tmpMat, 80.0, 90.0)
    return tmpMat
}


private var mediaPlayer: MediaPlayer? = null  // Declare the MediaPlayer instance globally

fun Mat.fireDetection(context: Context): Mat {
    val result = Mat()
    val hsv = Mat()
    val blur = Mat()
    val mask = Mat()

    Imgproc.GaussianBlur(this, blur, Size(11.0, 11.0), 0.0)
    Imgproc.cvtColor(blur, hsv, Imgproc.COLOR_BGR2HSV)

    val lower = Scalar(0.0, 73.0, 140.0)
    val upper = Scalar(255.0, 255.0, 255.0)
    Core.inRange(hsv, lower, upper, mask)

    Core.bitwise_and(this, this, result, mask)

    val contours = ArrayList<MatOfPoint>()
    Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

    // Flag to check if fire is currently detected
    var fireDetected = false

    for (contour in contours) {
        val boundingRect = Imgproc.boundingRect(contour)

        // Check if the detected region has a size between 30 and 150 pixels
        if (boundingRect.width in 30..150 && boundingRect.height in 30..150) {
            // Fire detected
            println("Fire Detected! Playing the alarm sound.")

            // Set the flag to true
            fireDetected = true

            // Play the alarm sound from assets
            playAlarmSound(context)

            // You can add additional logic here if needed
        }
    }

    // If fire is not detected, stop the alarm
    if (!fireDetected) {
        stopAlarm()
    }

    return result
}

private fun playAlarmSound(context: Context) {
    try {
        // Open a file descriptor for the alarm sound in assets
        val alarmFileDescriptor = context.assets.openFd("alarm.mp3")

        // Create a MediaPlayer instance if not already created
        if (mediaPlayer == null) {
            mediaPlayer = MediaPlayer()
        }

        // Set the data source from the file descriptor
        mediaPlayer?.setDataSource(
            alarmFileDescriptor.fileDescriptor,
            alarmFileDescriptor.startOffset,
            alarmFileDescriptor.length
        )

        // Prepare and start the MediaPlayer
        mediaPlayer?.prepare()
        mediaPlayer?.start()

        // Release the MediaPlayer resources when finished
        mediaPlayer?.setOnCompletionListener {
            stopAlarm()  // Stop the alarm when playback is completed
        }

    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun stopAlarm() {
    // Stop the alarm playback if the MediaPlayer instance is not null
    mediaPlayer?.stop()
    mediaPlayer?.release()
    mediaPlayer = null  // Set MediaPlayer instance to null after release
}

