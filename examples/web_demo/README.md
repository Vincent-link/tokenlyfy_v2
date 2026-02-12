# 加密投研助手 Web 演示

基于 Gradio 的 MVP 产品演示页面。

## 依赖

需安装 `evaluation` 可选依赖（含 Gradio、pandas、pandas-ta）：

```bash
uv pip install -e ".[evaluation]"
# 或
pip install -e ".[evaluation]"
```

若 technical 工具报错缺少 pandas/pandas-ta，可单独安装：

```bash
pip install pandas pandas-ta
```

## 运行

```bash
# 从项目根目录
uv run python examples/web_demo/gradio_demo.py
```

浏览器访问 http://127.0.0.1:7861

## 功能

- 个性化分析助手：先结论再四部分（价格位置/技术面/资金面/操作提示）
- 工具调用：crypto_price、technical、fear_greed、futures_data、search
- 记忆系统：session 匿名 ID 持久化，同一设备保留对话记忆
- 聊天历史：save_history 保存到浏览器 localStorage，关闭后重开可恢复
- 示例问题：点击即可快速体验
