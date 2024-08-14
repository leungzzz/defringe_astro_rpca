import os


def list_files_without_extension(directory, output_file):
    """
    读取目录中的所有文件名（不包括文件后缀），并存储在指定的txt文件中。

    :param directory: 目标目录的路径
    :param output_file: 输出的txt文件路径
    """
    try:
        # 获取指定目录中的所有文件名
        files = os.listdir(directory)

        with open(output_file, 'w') as file:
            for filename in files:
                # 获取文件名（去除后缀）
                name, _ = os.path.splitext(filename)
                # 写入txt文件，每行一个文件名
                file.write(name + '\n')

        print(f"文件名已成功写入 {output_file}")

    except Exception as e:
        print(f"出现错误: {e}")


# 示例用法
if __name__ == "__main__":
    # 设置目标目录路径
    directory = r"D:\作业批改和收集\信息系统分析与设计\课堂作业3（20240422）\学生答题"  # 替换为你的目标目录
    # 设置输出txt文件路径
    output_file = "filenames3.txt"

    list_files_without_extension(directory, output_file)
