# paddle node
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode?ref=badge_shield)

A third-party node.js addon for Paddle Lite.

Tested on Windows 10, Node.js v12.16.3.

Current features:

- Import model file generated from OPT tool.
- Infer from float array.
- Set thread number and power mode.

## Example

```javascript
var addon = require('./paddlenode');
addon.set_model_file('./mobilenetv1_opt.nb');
addon.set_threads(4);
addon.set_power_mode(2); //LITE_POWER_FULL
var arr = new Array(150528);
for(var i=0; i<arr.length; i++) arr[i]=1;
addon.infer_float(arr,[1, 3, 224, 224]);
```

The array returned is calculated from the model. But we manually add an element to indicate the size of the array, it's located at the first position.

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FKernelErr%2Fpaddlenode?ref=badge_large)