# defringe_astro_rpca
 use rpca to remove fringe pattern.

### 可处理的内容

1. 使用 PCP 方法进行defringe
2. CCD存储结构已经重新调整，所有待处理图片放在一个文件夹下，默认处理该文件夹下的所有图片
3. 配置 `config.py` 文件，需要调整的参数都放到该函数下。

### 使用
```python
python algorithms_rpca.py  # 直接运行该命令，需要调整的参数到 config.py 调整
