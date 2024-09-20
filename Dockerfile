# 使用官方的 Windows Server Core 镜像作为基础镜像
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# 安装 Python 3.10
RUN powershell -Command ` \
    $ErrorActionPreference = 'Stop'; ` \
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe -OutFile python-installer.exe; ` \
    Start-Process python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -NoNewWindow -Wait; ` \
    Remove-Item -Force python-installer.exe

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 文件复制到工作目录
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录下的所有文件复制到工作目录
COPY . .

# 暴露端口 8000
EXPOSE 8000

# 运行应用程序
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
