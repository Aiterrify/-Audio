# -*- coding: utf-8 -*-
"""安装脚本的配置生成部分：把模板里的 @MODELS@ 占位符替换成实际路径。

单独成文件，避免在 .bat 里嵌套多行 Python（cmd 对 heredoc 支持不可靠）。
"""
import json
import os
import sys


def main():
    # 本脚本位于 deploy/ 下，项目根目录是它的上一级
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src = os.path.join(here, "voices.runtime.json")
    dst = os.path.join(here, "voices.local.json")

    if not os.path.exists(src):
        print("  [ERROR] 模板不存在: %s" % src)
        return 1

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    models = os.path.join(root, "models").replace("\\", "/")
    for v in data.get("voices", []):
        for key in ("gpt", "sovits", "ref"):
            val = v.get(key)
            if isinstance(val, str):
                v[key] = val.replace("@MODELS@", models)
        for emo in v.get("emotions", {}).values():
            ref = emo.get("ref")
            if isinstance(ref, str):
                emo["ref"] = ref.replace("@MODELS@", models)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 顺手校验模型文件是否真的存在，缺哪个直接报出来，比启动时才报错友好
    missing = []
    for v in data.get("voices", []):
        for key in ("gpt", "sovits", "ref"):
            p = v.get(key)
            if isinstance(p, str) and not os.path.exists(p):
                missing.append((v.get("name", v.get("id")), key, p))

    print("  已生成 %s" % dst)
    if missing:
        print("  [WARN] 以下文件缺失，合成会失败：")
        for name, key, p in missing:
            print("         %s.%s -> %s" % (name, key, p))
        return 2
    print("  音色文件校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
