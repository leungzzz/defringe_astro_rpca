# defringe_astro_rpca
 use rpca to remove fringe pattern.

--- 

### ��������
1. ����ģ�黯�͹��ܷ��롣�� `images.py` �е����ݴ������� `algorithms_rpca.py` �е��㷨ʵ�ַֿ���ͼ���ȡ��Ԥ������� `ImageProcessor` ���У������㷨ʵ�ַ��� `FringeRemovalAlgorithm` ���С�


### �ɴ��������

1. ʹ�� PCP ��������defringe
2. CCD�洢�ṹ�Ѿ����µ��������д�����ͼƬ����һ���ļ����£�Ĭ�ϴ�����ļ����µ�����ͼƬ
3. ���� `config.py` �ļ�����Ҫ�����Ĳ������ŵ��ú����¡�

### ʹ��

˵��������֧ʹ�� CFHT �����ݽ��в��ԣ�����Ϻá�
```python
python algorithms_rpca.py  # ֱ�����и������Ҫ�����Ĳ����� config.py ����
