# utility-load-dataset
This module can be used for loading your original image datasets and convert them into NumPy arrays.
オリジナルのデータセットを読み込んで，numpy arrayに変換して返すモジュールです．

## 使い方

### データセットの準備
1. 適当なデータセットを自前で用意する．
2. 拡張子をjpeg, png, tiff, bmp, pgmのいずれかに設定する．
3. ラベリングをしたければ，カテゴリ別にフォルダを作成し，分類する．

#### ラベリングの仕方について
ラベリングをする際のフォルダ名は，  
"category_0", "category_1", ...  
のように，"category" + "_" + "半角数字"の形式で入力をすると，区別されます．

### データセットのロード方法
keras.datasetsから任意のデータセットをimportするのと同様の手続きで使うことができます．
1. 下準備を行う
```sh
# モジュールの呼び出し
from utils import load_dataset
# 画像サイズの設定
img_size = (28, 28, 1)
# 読み込むフォルダの指定
folders = [
		'category_0',
		'category_1',
		'category_2',
		'category_3',
		'category_4',
		'category_5',
		'category_6',
		'category_7',
		'category_8',
		'category_9'
	]
```
2. データセットをロードする

```sh
(x_train, y_train), (x_test, y_test) = load_dataset(root_dir=r'/Users/gucci/Downloads/jaffe',
                                  folders=folders, test_size=0.10, img_size=img_size)
```
* `root_dir`にはデータセットが置かれているディレクトリのパスを指定します．
* `folders`には読み込むフォルダを指定したリストを渡します．
* `test_size`にはデータセットをどの比率(train:test)で分割するかを指定します．
* `img_size`にはデータセットとしての画像サイズを指定します．入力画像は必要に応じて自動的にリサイズされます．

## テスト
[畳み込み版VAE](https://github.com/gucci-j/conv-vae)を基に，データセットに[JAFFE](http://www.kasrl.org/jaffe.html)を使用してテストをしました．  
ソースコードは[test.py](https://github.com/gucci-j/utility-load-dataset/blob/master/test.py)を参照してください．  
* 1000 epochsでの出力結果
![動作結果](https://user-images.githubusercontent.com/30075338/38455242-4996e6b4-3ab0-11e8-958d-f822634f8265.png)
