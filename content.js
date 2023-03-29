   const FLOWER_CLASS = {
      0: "daisy",
      1: "lotus",
      2: "rose",
      3: "sunflower",
      4: "tulip",
    };

    // Load model
    $("document").ready(async function () {
      model = await tf.loadLayersModel("./models/tfjs_model/model.json");
      console.log("Load model");
      console.log(model.summary());
    });

    $("#upload_button").click(function () {
      $("#fileinput").trigger("click");
    });

    async function predict() {
      // 1. Chuyen anh ve tensor
      let image = document.getElementById("display_image");
      let img = tf.browser.fromPixels(image);
      let normalizationOffset = tf.scalar(255 / 2); // 127.5
      let tensor = img
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .sub(normalizationOffset)
        .div(normalizationOffset)
        .reverse(2)
        .expandDims();

      // 2. Predict
      let predictions = await model.predict(tensor);
      predictions = predictions.dataSync();
      console.log(predictions);

      // 3. Hien thi len man hinh
      let top5 = Array.from(predictions)
        .map(function (p, i) {
          return {
            probability: p,
            className: FLOWER_CLASS[i],
          };
        })
        .sort(function (a, b) {
          return b.probability - a.probability;
        });
      console.log(top5);
      $("#result_info").empty();
      top5.forEach(function (p) {
        $("#result_info").append(
          `<li>${p.className}: ${p.probability.toFixed(3)}</li>`
        );
      });
    }

    $("#fileinput").change(function () {
      let reader = new FileReader();
      reader.onload = function () {
        let dataURL = reader.result;

        imEl = document.getElementById("display_image");
        imEl.onload = function () {
          predict();
        };
        $("#display_image").attr("src", dataURL);
        $("#result_info").empty();
      };

      let file = $("#fileinput").prop("files")[0];
      reader.readAsDataURL(file);
    });