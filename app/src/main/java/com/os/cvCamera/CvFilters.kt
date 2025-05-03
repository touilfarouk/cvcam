package com.os.cvCamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaPlayer
import android.util.Log
import android.widget.Toast
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.CvType.CV_8UC1
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
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

private var mediaPlayer: MediaPlayer? = null

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

    var fireDetected = false

    for (contour in contours) {
        val boundingRect = Imgproc.boundingRect(contour)
        if (boundingRect.width in 30..150 && boundingRect.height in 30..150) {
            println("Fire Detected! Playing the alarm sound.")
            fireDetected = true
            playAlarmSound(context)
        }
    }

    if (!fireDetected) {
        stopAlarm()
    }

    return result
}

private fun playAlarmSound(context: Context) {
    try {
        val alarmFileDescriptor = context.assets.openFd("alarm.mp3")
        if (mediaPlayer == null) {
            mediaPlayer = MediaPlayer()
        }
        mediaPlayer?.setDataSource(
            alarmFileDescriptor.fileDescriptor,
            alarmFileDescriptor.startOffset,
            alarmFileDescriptor.length
        )
        mediaPlayer?.prepare()
        mediaPlayer?.start()
        mediaPlayer?.setOnCompletionListener {
            stopAlarm()
        }

    } catch (e: Exception) {
        e.printStackTrace()
    }
}

private fun stopAlarm() {
    mediaPlayer?.stop()
    mediaPlayer?.release()
    mediaPlayer = null
}

/**
 * Detects ID number text from a card image using ML Kit Text Recognition
 */

fun Mat.idScan(context: Context, onResult: (String?) -> Unit) {
    val bitmap = Bitmap.createBitmap(this.cols(), this.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(this, bitmap)

    val image = InputImage.fromBitmap(bitmap, 0)
    val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    recognizer.process(image)
        .addOnSuccessListener { visionText ->
            var idNumber: String? = null
            var boundingBox: android.graphics.Rect? = null

            // Accumulate all numeric text blocks (concatenated digits)
            val digitSequences = mutableListOf<Pair<String, android.graphics.Rect?>>()

            for (block in visionText.textBlocks) {
                for (line in block.lines) {
                    val digitsInLine = line.text.replace("[^\\d]".toRegex(), "") // remove non-digits
                    if (digitsInLine.length >= 14) { // heuristic: likely part of ID number
                        digitSequences.add(Pair(digitsInLine, line.boundingBox))
                    }
                }
            }

            // Combine all sequences and check if we get 18 digits
            val combinedDigits = digitSequences.joinToString("") { it.first }

            if (combinedDigits.length == 18) {
                idNumber = combinedDigits

                // Optionally: merge bounding boxes (or just take first one)
                boundingBox = digitSequences.firstOrNull()?.second

                // Draw green rectangle
                boundingBox?.let {
                    val canvas = Canvas(bitmap)
                    val paint = Paint().apply {
                        color = Color.GREEN
                        strokeWidth = 5f
                        style = Paint.Style.STROKE
                    }
                    canvas.drawRect(it, paint)
                }

                Toast.makeText(context, "ID Number Detected: $idNumber", Toast.LENGTH_LONG).show()
                Log.d("IDScan", "Detected ID Number: $idNumber")
            } else {
                Toast.makeText(context, "No valid 18-digit ID number detected", Toast.LENGTH_SHORT).show()
                Log.d("IDScan", "No valid 18-digit ID number detected")
            }

            onResult(idNumber)
        }
        .addOnFailureListener { e ->
            e.printStackTrace()
            Toast.makeText(context, "Text recognition failed", Toast.LENGTH_SHORT).show()
            onResult(null)
        }
}
