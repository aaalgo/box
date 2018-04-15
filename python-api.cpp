#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
using namespace boost::python;
namespace np = boost::python::numpy;

namespace {
    using std::istringstream;
    using std::ostringstream;
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::vector;

    class AlignBoxes {
    public:
        list apply (np::ndarray gt_boxes, np::ndarray boxes) {
            CHECK(gt_boxes.get_nd() == 2);
            CHECK(boxes.get_nd() == 2);
            CHECK(gt_boxes.shape(1) >= 3);
            CHECK(boxes.shape(1) == 4);
            list result;
            return result;
        }
    };

    class MaskExtractor {
        cv::Size sz;
    public:
        MaskExtractor (int width, int height): sz(width, height) {
        }

        np::ndarray apply (np::ndarray images,
                    np::ndarray gt_boxes,
                    np::ndarray boxes) {
            CHECK(images.get_nd() == 4);
            CHECK(gt_boxes.get_nd() == 2);
            CHECK(boxes.get_nd() == 2);
            CHECK(gt_boxes.shape(1) >= 3);
            CHECK(boxes.shape(1) == 4);
            CHECK(gt_boxes.shape(0) == boxes.shape(0));
            int n = gt_boxes.shape(0);
            int H = images.shape(1);
            int W = images.shape(2);
            int C = images.shape(3);
            CHECK(C == 1);

            np::ndarray masks = np::zeros(make_tuple(n, sz.height, sz.width, 1), np::dtype::get_builtin<float>());

            for (int i = 0; i < n; ++i) {
                float *gt_box = (float *)(gt_boxes.get_data() + i * gt_boxes.strides(0));
                float *box = (float *)(boxes.get_data() + i * boxes.strides(0));
                int index(gt_box[0]);
                int tag(gt_box[2]);
                cv::Mat image(H, W, CV_32F, images.get_data() + index * images.strides(0));
                float *mask_begin =  (float *)(masks.get_data() + i * masks.strides(0));
                float *mask_end = mask_begin + masks.strides(0);
                cv::Mat mask(sz, CV_32F, mask_begin);

                int x1 = int(round(box[0]));
                int y1 = int(round(box[1]));
                int x2 = int(round(box[2]));
                int y2 = int(round(box[3]));
                CHECK(x1 >= 0);
                CHECK(y1 >= 0);
                CHECK(x2 < W);
                CHECK(y2 < H);
                cv::Mat from(image(cv::Rect(x1, y1, x2-x1, y2-y1)));
                cv::resize(from, mask, mask.size(), 0, 0, CV_INTER_NN);

                for (float *p = mask_begin; p < mask_end; ++p) {
                    if (p[0] == tag) p[0] = 1.0;
                    else p[0] = 0.0;
                }
            }
            return masks;
        }
    };
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    class_<AlignBoxes>("AlignBoxes", init<>())
        .def("apply", &AlignBoxes::apply)
    ;
    class_<MaskExtractor>("MaskExtractor", init<int, int>())
        .def("apply", &MaskExtractor::apply)
    ;
}

