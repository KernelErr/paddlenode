# paddle node
A third-party node.js addon for Paddle Lite.

Tested on Windows 10, Node.js v12.16.3.

Current features:

- Import model file generated from OPT tool.
- Infer from float array.
- Set thread number and power mode.

## Example

```javascript
var addon = require('./paddlenode');
addon.set_model_file("./mobilenetv1_opt.nb");
addon.set_threads(4);
addon.set_power_mode(2); //LITE_POWER_FULL
var arr = new Array(150528);
for(var i=0; i<arr.length; i++) arr[i]=1;
addon.infer_float(arr,[1, 3, 224, 224]);
```

