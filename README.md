
## Install venv
```shell 
sudo apt install python3.10-venv
```

```shell
pip3 install pyautogui
pip3 install imageio
pip3 install selenium
```

```shell
sudo apt-get install python3-tk python3-dev
```

```shell
sudo apt install gnome-screenshot
```

## Install Chrome on Linux

##### Download and install the Google Chrome signing key
```shell
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
```

##### Add the Google Chrome repository to the system
```shell
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
```

##### Update the package list
```shell
sudo apt update
```

```shell
sudo apt install -y google-chrome-stable
```


##### Download GeckoDriver
https://github.com/mozilla/geckodriver/releases