# paddle node
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode?ref=badge_shield)

A third-party node.js addon for Paddle Inference. The beginning two versions of paddle node are using Paddle Lite for backbone, but we use Paddle Inference instead as it is more compatible for x86_64 platform.

Tested on Windows 10, Node.js v12.16.3.

Current features:

- Import combined model.
- Infer from float array.
- Image classification from file.

## Methods

#### set_combined_model

```javascript
paddlenode.set_combined_model(ModelDir)
```

#### infer_float

```javascript
paddlenode.infer_float(Data, Size)
```

#### image_file_classification

```javascript
paddlenode.image_file_classification(ImagePath, InputSize, Scalefactor, Mean, swapRB)
```

## Example

### Image Classification

```javascript
var paddlenode = require('./paddlenode');
paddlenode.set_combined_model('./mobilenetv1');
var res = paddlenode.image_file_classfication("test.jpg",[1, 3, 224, 224],0.007843,[224,224],[0.485,0.456,0.406], false)
console.log('Result', res.indexOf(Math.max(...res)));
```

### Sample Vector

```javascript
var paddlenode = require('./paddlenode');
paddlenode.set_combined_model('./mobilenetv1');
var arr = new Array(150528);
for(var i=0; i<arr.length; i++) arr[i]=1;
paddlenode.infer_float(arr,[1, 3, 224, 224]);
```

The array returned is calculated from the model. But we manually add an element to indicate the size of the array, it's located at the first position.

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode?ref=badge_large)