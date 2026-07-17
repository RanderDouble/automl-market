# 将 RQ1 / RQ3 页面并入 ZBY 主幻灯片

`C3_RQ1_RQ3.tex` 是独立预览入口，使用与 `C3_Paper_Theory.tex` 相同的 `zju_beamer.sty`、背景、页眉和页脚。

独立编译：

```bash
cd slides/zby
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=build C3_RQ1_RQ3.tex
```

若要并入 ZBY 的主文件，在其导言区把图片路径改为：

```tex
\graphicspath{{./figures/}{../../results/figures/}{../../docs/assets/}}
```

然后在理论部分结束、附录开始之前加入：

```tex
\input{../rq1_rq3_slides.tex}
```

`rq1_rq3_slides.tex` 共 13 个 frame：前 10 个是正式内容，最后 3 个位于“备查页”章节。若总时长较短，可以只保留正式内容；不要删除备查源码，老师提问时可直接跳转使用。
