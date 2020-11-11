
class Main {
  constructor() {
    // Initialize images
    this.contentImg = document.getElementById('content-img');
    this.styleImg = document.getElementById('style-img');
    this.setImage(this.contentImg, 'fileSelect-1');
    this.setImage(this.styleImg, 'fileSelect-2');

    // Initialize buttons
    this.styleButton = document.getElementById('style-button');
    this.styleButton.onclick = () => {
      this.disableStylizeButtons();
      this.startStyling().finally(() => {
        this.enableStylizeButtons();
      });
    };
    // initializeStyleTransfer
    this.stylized = document.getElementById('stylized');
    Promise.all([
      this.loadMobileNetStyleModel(),
      this.loadSeparableTransformerModel(),
    ]).then(([styleNet, transformNet]) => {
      console.log('Loaded styleNet');
      this.styleNet = styleNet;
      this.transformNet = transformNet;
      this.enableStylizeButtons()
    });
  }
  /**
 * setImage
 */
  // Helper function for setting an image
  setImage(element, fileId) {
    const fileSelect = document.getElementById(fileId);
    fileSelect.onchange = (evt) => {
      const f = evt.target.files[0];
      const fileReader = new FileReader();
      fileReader.onload = ((e) => {
        element.src = e.target.result;
      });
      fileReader.readAsDataURL(f);
      fileSelect.value = '';
    }
  }

  /**
   * disableStylizeButtons
   */
  disableStylizeButtons() {
    this.styleButton.disabled = true;
    // modelSelectStyle.disabled = true;
    // modelSelectTransformer.disabled = true;
  }
  /**
   * loadMobileNetStyleModel
   */
  async loadMobileNetStyleModel() {
    if (!this.mobileStyleNet) {
      this.mobileStyleNet = await tf.loadGraphModel(
        'model/saved_model_style_js/model.json');
    }
    return this.mobileStyleNet;
  }
  /**
   * loadSeparableTransformerModel
   */
  async loadSeparableTransformerModel() {
    if (!this.separableTransformNet) {
      this.separableTransformNet = await tf.loadGraphModel(
        'model/saved_model_transformer_separable_js/model.json'
      );
    }

    return this.separableTransformNet;
  }
  /**
   * enableStylizeButtons
   */
  enableStylizeButtons() {
    this.styleButton.disabled = false;
    this.styleButton.textContent = 'Stylize';
  }


  /**
   * startStyling
   */
  async startStyling() {
    await tf.nextFrame();
    this.styleButton.textContent = 'Generating 100D style representation';
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
    })
    this.styleButton.textContent = 'Stylizing image...';
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
      return this.transformNet.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
    })
    await tf.browser.toPixels(stylized, this.stylized);
    bottleneck.dispose();  // Might wanna keep this around
    stylized.dispose();
    // show result
    document.getElementById('resultCanvas').classList.remove('d-none');
  }

}

window.mobilecheck = function () {
  var check = false;
  (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
  return check;
};
window.addEventListener('load', () => new Main());

// /**
//  * setImage
//  */
// // Helper function for setting an image
// function setImage(element, fileId) {
//   const fileSelect = document.getElementById(fileId);
//   fileSelect.onchange = (evt) => {
//     const f = evt.target.files[0];
//     const fileReader = new FileReader();
//     fileReader.onload = ((e) => {
//       element.src = e.target.result;
//     });
//     fileReader.readAsDataURL(f);
//     fileSelect.value = '';
//   }
// }

// // Initialize images
// const contentImg = document.getElementById('content-img');
// const styleImg = document.getElementById('style-img');
// setImage(contentImg, 'fileSelect-1');
// setImage(styleImg, 'fileSelect-2');

// // Initialize buttons
// const styleButton = document.getElementById('style-button');
// styleButton.onclick = () => {
//   this.disableStylizeButtons();
//   this.startStyling().finally(() => {
//     this.enableStylizeButtons();
//   });
// };
// Promise.all([
//   this.loadMobileNetStyleModel(),
//   this.loadSeparableTransformerModel(),
// ]).then(([styleNet, transformNet]) => {
//   console.log('Loaded styleNet');
//   this.styleNet = styleNet;
//   this.transformNet = transformNet;
//   this.enableStylizeButtons()
// });

// /**
//  * disableStylizeButtons
//  */
// function disableStylizeButtons() {
//   styleButton.disabled = true;
//   // modelSelectStyle.disabled = true;
//   // modelSelectTransformer.disabled = true;
// }
// /**
//  * loadMobileNetStyleModel
//  */
// async function loadMobileNetStyleModel() {
//   if (!this.mobileStyleNet) {
//     this.mobileStyleNet = await tf.loadGraphModel(
//       'saved_model_style_js/model.json');
//   }
//   return this.mobileStyleNet;
// }
// /**
//  * loadSeparableTransformerModel
//  */
// async function loadSeparableTransformerModel() {
//   if (!this.separableTransformNet) {
//     this.separableTransformNet = await tf.loadGraphModel(
//       'saved_model_transformer_separable_js/model.json'
//     );
//   }

//   return this.separableTransformNet;
// }
// /**
//  * enableStylizeButtons
//  */
// function enableStylizeButtons() {
//   styleButton.disabled = false;
//   styleButton.textContent = 'Stylize';
// }


// /**
//  * startStyling
//  */
// async function startStyling() {
//   await tf.nextFrame();
//   styleButton.textContent = 'Generating 100D style representation';
//   await tf.nextFrame();
//   let bottleneck = await tf.tidy(() => {
//     return styleNet.predict(tf.browser.fromPixels(styleImg).toFloat().div(tf.scalar(255)).expandDims());
//   })
//   styleButton.textContent = 'Stylizing image...';
//   await tf.nextFrame();
//   const stylized = await tf.tidy(() => {
//     return transformNet.predict([tf.browser.fromPixels(contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
//   })
//   await tf.browser.toPixels(stylized, this.stylized);
//   console.log(stylized)
//   bottleneck.dispose();  // Might wanna keep this around
//   stylized.dispose();
// }