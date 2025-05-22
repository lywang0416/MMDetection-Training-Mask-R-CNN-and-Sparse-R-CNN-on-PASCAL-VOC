from mmdet.datasets import VOCDataset
from mmdet.registry import DATASETS
import os.path as osp
import xml.etree.ElementTree as ET

@DATASETS.register_module()
class CustomVOCDataset(VOCDataset):
    def __init__(self, **kwargs):
        # 确保 data_prefix 包含所有必要的子路径
        if 'data_prefix' in kwargs:
            prefix = kwargs['data_prefix']
            # 如果只提供了 sub_data_root，补充其他路径
            if 'sub_data_root' in prefix:
                prefix['img'] = osp.join(prefix['sub_data_root'], 'JPEGImages')
                prefix['ann'] = osp.join(prefix['sub_data_root'], 'Annotations')
                prefix['seg'] = osp.join(prefix['sub_data_root'], 'SegmentationObject')
            kwargs['data_prefix'] = prefix
        super().__init__(**kwargs)

    def parse_data_info(self, raw_img_info):
        """解析图像和标注信息"""
        img_id = raw_img_info['img_id']

        # 使用 data_prefix 中的路径信息
        img_path = osp.join(self.data_prefix['img'], f'{img_id}.jpg')
        xml_path = osp.join(self.data_prefix['ann'], f'{img_id}.xml')

        # 验证文件是否存在
        if not osp.exists(xml_path):
            raise FileNotFoundError(f"XML 文件不存在: {xml_path}")

        # 解析 XML 文件
        data_info = {}
        data_info['img_path'] = img_path
        data_info['img_id'] = img_id

        # 读取 XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise RuntimeError(f"解析 XML 文件失败: {xml_path}") from e

        # 解析图像尺寸 (修复 KeyError: 'width' 的问题)
        size = root.find('size')
        if size is not None:
            width = size.find('width')
            height = size.find('height')
            if width is not None and height is not None:
                data_info['width'] = int(width.text)
                data_info['height'] = int(height.text)
            else:
                # 如果 XML 中没有 width/height，尝试从图像文件获取
                import cv2
                img = cv2.imread(img_path)
                if img is not None:
                    data_info['height'], data_info['width'] = img.shape[:2]
                else:
                    raise ValueError(f"无法读取图像: {img_path}")
        else:
            # 如果没有 size 标签，尝试从图像文件获取
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                data_info['height'], data_info['width'] = img.shape[:2]
            else:
                raise ValueError(f"无法读取图像: {img_path}")

        # 解析标注信息
        if self.test_mode:
            data_info['instances'] = []
        else:
            # 检查是否有分割标注
            segmented = int(root.find('segmented').text) if root.find('segmented') is not None else 0
            data_info['has_seg'] = segmented == 1

            # 解析边界框和类别
            instances = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)

                difficult = obj.find('difficult')
                difficult = int(difficult.text) if difficult is not None else 0
                instances.append(
                    dict(
                        bbox=[x1, y1, x2, y2],
                        bbox_label=self.cat2label[name],
                        mask=osp.join(self.data_prefix['seg'], f'{img_id}.png') if segmented else None,
                        ignore_flag=difficult
                    )
                )
            data_info['instances'] = instances

        return data_info