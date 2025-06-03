import nltk
import ssl
import sys

print("开始下载NLTK数据包...")

# 处理SSL证书问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载必要的数据包
required_packages = [
    'punkt',           # 分词器
    'stopwords',       # 停用词
    'punkt_tab'        # punkt表格数据
]

success = True
for package in required_packages:
    try:
        print(f"正在下载 {package}...")
        nltk.download(package)
        print(f"{package} 下载成功!")
    except Exception as e:
        print(f"下载 {package} 时出错: {e}")
        success = False

if success:
    print("\n所有NLTK数据包下载成功!")
else:
    print("\n部分NLTK数据包下载失败，请查看上面的错误信息。")
    sys.exit(1)