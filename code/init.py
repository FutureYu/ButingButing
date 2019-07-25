from rpi_define import *
with open(BUTING_PATH + r"\model\checkpoint", "r") as f:
    contents = f.read().split("\n")
    for i, content in enumerate(contents):
        res = contents[i].split(r"BUTING_PATH")
        res = res[0] + BUTING_PATH.replace("\\", "\\\\") + res[1]
        contents[i] = res

with open(BUTING_PATH + r"\model\checkpoint", "w") as f:
    for content in contents:
        f.write(content + "\n")