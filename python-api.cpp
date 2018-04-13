#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
#include <opencv2/opencv.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <glog/logging.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/ndarrayobject.h>
using namespace boost::python;

namespace {
    using std::istringstream;
    using std::ostringstream;
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::vector;

    struct Circle {

        static unsigned constexpr PARAMS= 3;

        struct Shape {
            float x, y, r;
        };

        static void update_shape (Shape *s, cv::Point_<float> const &pt, float const *params) {
            s->x = pt.x + params[0];
            s->y = pt.y + params[1];
            s->r = params[2];

        }
        static void update_params (Shape const &c, float *params) {
            params[0] = c.x;
            params[1] = c.y;
            params[2] = c.r;
        }

        static float overlap (Shape const &a, Shape const &b) {
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float d = sqrt(dx * dx + dy * dy);
            float r = std::max(a.r, b.r) + 1;
            return  (r-d)/r;
        }

        static void draw (cv::Mat image, Shape const &c) {
            int r = std::round(c.r);
            int x = std::round(c.x);
            int y = std::round(c.y);
            cv::circle(image, cv::Point(x, y), r, cv::Scalar(255, 0, 0), 1);
        }
    };

    struct Box {
        static unsigned constexpr PARAMS= 4;

        typedef cv::Rect_<float> Shape;

        static void update_shape (Shape *s, cv::Point_<float> const &pt, float const *params) {
            s->x = pt.x + params[0] - params[2]/2;
            s->y = pt.y + params[1] - params[3]/2;
            s->width = params[2];
            s->height = params[3];
        }

        static float overlap (Shape const &s1, Shape const &s2) {
            float o = (s1 & s2).area();
            return o / (s1.area() + s2.area() - o +1);
        }

        static void update_params (Shape const &s, float *params) {
            params[0] = s.x;
            params[1] = s.y;
            params[2] = s.x + s.width;
            params[3] = s.y + s.height;
        }

        static void draw (cv::Mat image, Shape const &c) {
            cv::rectangle(image, cv::Point(int(round(c.x)), int(round(c.y))),
                                 cv::Point(int(round(c.x+c.width)), int(round(c.y+c.height))), 
                                 cv::Scalar(255, 0, 0), 1);
        }
    };

    template <typename SHAPE>
    class ShapeProposal {
        int upsize;
        float pth;
        float th;

        struct Shape: public SHAPE::Shape {
            float score;
            float keep;
        };
    public:
        ShapeProposal (int up, float pth_, float th_): upsize(up), pth(pth_), th(th_) {
        }

        PyObject* apply (PyObject *prob_, PyObject *params_, PyObject *image_) {
            cv::Mat prob(pbcvt::fromNDArrayToMat(prob_));
            cv::Mat params(pbcvt::fromNDArrayToMat(params_));

            CHECK(prob.type() == CV_32F);
            //CHECK(params.type() == CV_32FC3);
            CHECK(prob.rows == params.rows);
            CHECK(prob.cols == params.cols);
            //CHECK(prob.channels() == 1);
            CHECK(params.channels() == SHAPE::PARAMS * prob.channels());
            vector<Shape> all;
            int priors = prob.channels();
            for (int y = 0; y < prob.rows; ++y) {
                float const *pl = prob.ptr<float const>(y);
                float const *pp = params.ptr<float const>(y);
                for (int x = 0; x < prob.cols; ++x) {
                    cv::Point_<float> pt(x * upsize, y * upsize);
                    for (int prior = 0; prior < priors; ++prior, ++pl, pp += SHAPE::PARAMS) {
                        if (pl[0] < pth) continue;
                        Shape c;
                        SHAPE::update_shape(&c, pt, pp);
                        c.score = pl[0];
                        c.keep = true;
                        all.push_back(c);
                    }
                }
            }
            sort(all.begin(), all.end(), [](Shape const &a, Shape const &b){return a.score > b.score;});

            unsigned cnt = 0;
            for (unsigned i = 0; i < all.size(); ++i) {
                if (!all[i].keep) continue;
                cnt += 1;
                Shape const &a = all[i];
                for (unsigned j = i+1; j < all.size(); ++j) {
                    Shape &b = all[j];
                    float d = SHAPE::overlap(a, b);
                    if (d > th) {
                       b.keep = false;
                    }
                }
            }
            cv::Mat result(cnt, SHAPE::PARAMS, CV_32F);
            int next = 0;
            for (auto const &c: all) {
                if (!c.keep) continue;
                float *r = result.ptr<float>(next++);
                SHAPE::update_params(c, r);
            }
            CHECK(next == cnt);

            if (image_ != Py_None) {
                cv::Mat image = pbcvt::fromNDArrayToMat(image_);
                for (auto const &c: all) {
                    if (!c.keep) continue;
                    SHAPE::draw(image, c);
                }
            }
            return pbcvt::fromMatToNDArray(result);
        }
    };
}

int init_numpy()
{
    import_array();
    return 0;
}

BOOST_PYTHON_MODULE(cpp)
{
	init_numpy();
    scope().attr("__doc__") = "adsb4";
    to_python_converter<cv::Mat,
                     pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();
    class_<ShapeProposal<Circle>>("CircleProposal", init<int, float, float>())
        .def("apply", &ShapeProposal<Circle>::apply)
    ;
    class_<ShapeProposal<Box>>("BoxProposal", init<int, float, float>())
        .def("apply", &ShapeProposal<Box>::apply)
    ;
}

