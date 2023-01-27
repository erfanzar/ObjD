
# ObjD 🚀
smart and strong object detection ai built from scatch with some new features 🚀🚀 (Using some custom CNNs and NeuralNets with a holy help from darknet and yolo for everysingle part and adding a bit of research to it 😂)


![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Training

To train ObjD you don't have to do crazy things you just have to pick one of the models and train that with custom data that you want to train ObjD with :)


Help For Training 

*ArgParses*


`cfg`
```bash
  python3 torch_train.py --cfg <config/config.yaml>
```

`data`
```bash
  python3 torch_train.py --data <data/path.yaml>
```


`epochs`
```bash
  python3 torch_train.py --epochs <300>
```


`eval`
```bash
  python3 torch_train.py --eval
```


`debug`
```bash
  python3 torch_train.py --debug
```



`device`
```bash
  python3 torch_train.py --device <'cuda:0'>
```



`auto anchors`
```bash
  python3 torch_train.py --auto-anchors
```
## Model Reference (ObjD)

| Model             | Param     |  Accuracy | FPS|                           
| ----------------- | -----------|- | -|
| ObjD tiny | `~ M` |accuracy on COCO data 50 %| `FPS 80`|
| ObjD s | `~ M` |accuracy on COCO data 59 %| `FPS 72`|
| ObjD n | `~ M` |accuracy on COCO data 68 %| `FPS 60`|
| ObjD hx | `~ M`|accuracy on COCO data 77 %| `FPS 43`|



## Types

#### About 

there are 2 types of training methods 
 

- [PytorchLightning](https://github.com/erfanzar/ObjD/blob/main/train.py)
- [Native Pytorch](https://github.com/erfanzar/ObjD/blob/main/torch_train.py)

they both have some benefits and the both are hackable to make them more customize and its depends on you to choose which way you want to train your model with but i recommend yo use native pytorch cause that one have better work and stabelity right now

#### PytorchLightning 

```bash
  python3 train.py 
```

#### Native Pytorch

```python
    python3 torch_train.py 
```





## 🚀 About Me
Hi there 👋
I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Used By

This project is used by the following companies:

- You Can Be First One Here :)



## Author

- [@erfanzar](https://www.github.com/erfanzar)

