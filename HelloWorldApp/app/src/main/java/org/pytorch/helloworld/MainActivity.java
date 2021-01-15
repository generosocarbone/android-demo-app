package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Calendar;
import java.util.Locale;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  private static final String TAG = "PytorchHelloWorld";

  private String[] assets = new String[]{
          "image.jpg",
          "colibri.jpg",
//          "vaso.jpg",
          "vaso_2000x2667.jpg",
          "vaso_1000x1334.jpg",
  };

  private TextView textView;
  private TextView metadata;
  private Module module;
  private ImageView imageView;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);


    module = null;
    String name = "colibri.jpg";
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg

      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = Module.load(assetFilePath(this, "model.pt"));
    } catch (IOException e) {
      Log.e(TAG, "Error reading assets", e);
//      finish();
      return;
    }

    // showing image on UI
    imageView = findViewById(R.id.image);
    textView = findViewById(R.id.text);

    handler.post(runnable);
  }


  private void classificateImage(String imageName) {
    if (module == null) {
      textView.setText(R.string.no_module_error);
    }

    Log.d(TAG, "classificateImage: image: " + imageName);

    Bitmap bitmap = null;
    try {
      bitmap = BitmapFactory.decodeStream(getAssets().open(imageName));
    } catch (Exception e) {
      metadata.setText(String.format("Cannot open image: %s", imageName));
      Log.e(TAG, String.format("Cannot open image: %s", imageName), e);
    }

    Log.d(TAG, "classificateImage: bitmap caricata");

    if (module == null || bitmap == null)
      return;

    try {
      imageView.setImageBitmap(bitmap);
    } catch (Exception e) {
      Log.e(TAG, String.format("Cannot display image. Classification still in progress: %s", imageName), e);
      Toast.makeText(this, "Cannot display image. Classification still in progress",Toast.LENGTH_LONG).show();
    }

    long start = Calendar.getInstance().getTimeInMillis();
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    // running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();
    long end = Calendar.getInstance().getTimeInMillis();
    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
    textView.setText(className);

    metadata = findViewById(R.id.metadata);
    metadata.setText(
            String.format(
                    Locale.ITALIAN,
                    "Image: %s\nTime: %dms\n%dx%d; %.02fMB; Score: %.02f: Class: %d",
                    imageName,
                    end - start,
                    bitmap.getWidth(),
                    bitmap.getHeight(),
                    (float)(bitmap.getByteCount() / 1024) / 1024,
                    maxScore,
                    maxScoreIdx
            )
    );
  }


  private Handler handler = new Handler(Looper.getMainLooper());
  private int index = 0;
  private Runnable runnable = new Runnable() {
    @Override
    public void run() {
      int nextIndex = (index + 1) % assets.length;

      Log.d(TAG, String.format(
              Locale.ITALIAN,
              "index: %d; size: %d; nextIndex: %d; image: %s; nextImage: %s",
              index,
              assets.length,
              nextIndex,
              assets[index],
              assets[nextIndex]
      ));

      textView.setText(String.format("Classifing image %s", assets[index]));

      classificateImage(assets[index]);

      index = (index + 1) % assets.length;

      handler.postDelayed(this, 5000);
    }
  };

  @Override
  protected void onPause() {
    super.onPause();
    handler.removeCallbacks(runnable);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
