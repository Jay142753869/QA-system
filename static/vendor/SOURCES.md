# Vendor 资源清单

本目录包含从 CDN 下载并本地化的前端依赖文件，用于离线/打包环境。

## 文件列表与来源

| 文件 | 版本 | 来源 URL |
|------|------|----------|
| `css/bootstrap.min.css` | 5.3.1 | https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/5.3.1/css/bootstrap.min.css |
| `js/bootstrap.bundle.min.js` | 5.3.1 | https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/5.3.1/js/bootstrap.bundle.min.js |
| `css/all.min.css` | 6.4.2 | https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.2/css/all.min.css |
| `js/jquery.min.js` | 3.7.1 | https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.min.js |
| `webfonts/fa-solid-900.woff2` | 6.4.2 | https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.2/webfonts/fa-solid-900.woff2 |
| `webfonts/fa-solid-900.ttf` | 6.4.2 | https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.2/webfonts/fa-solid-900.ttf |
| `webfonts/fa-regular-400.woff2` | 6.4.2 | https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.2/webfonts/fa-regular-400.woff2 |
| `webfonts/fa-brands-400.woff2` | 6.4.2 | https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.2/webfonts/fa-brands-400.woff2 |

## 许可证

- **Bootstrap**: MIT License (https://github.com/twbs/bootstrap/blob/main/LICENSE)
- **Font Awesome**: Font Awesome Free License (https://fontawesome.com/license/free)
- **jQuery**: MIT License (https://jquery.org/license/)

## 更新方式

如需更新版本，修改 `templates/index.html` 中的引用路径并重新下载对应版本文件到本目录。
