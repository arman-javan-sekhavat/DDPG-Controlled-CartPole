#include <mlpack/prereqs.hpp>
#include <mlpack.hpp>
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <GLFW/glfw3.h>

#include "Header.h"

using namespace mlpack;
using namespace arma;
using namespace ens;



// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;



void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {

    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}



void mouse_button(GLFWwindow* window, int button, int act, int mods) {

    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}


void mouse_move(GLFWwindow* window, double xpos, double ypos) {

    if (!button_left && !button_middle && !button_right) {
        return;
    }

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;


    int width, height;
    glfwGetWindowSize(window, &width, &height);


    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) == GLFW_PRESS;
    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;


    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}



void scroll(GLFWwindow* window, double xoffset, double yoffset) {

    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}




extern mjtNum force_scale;

extern FFN<EmptyLoss, GaussianInitialization>* policy = nullptr;
mat S(4, 1);
mat A(1, 1);

double Kp = 0.1;
double Ki = 0.1;
double e = 0.0;
double ie = 0.0;
double xd = 0.0;

void test_controller(const mjModel* m, mjData* d) {

    S(0, 0) = d->qpos[0] - xd;
    S(1, 0) = d->qvel[0];
    S(2, 0) = d->qpos[1];
    S(3, 0) = d->qvel[1];

    policy->Predict(S, A);
    d->ctrl[0] = force_scale*(A(0, 0));
}



int main()
{

    train();

    //--------------------------------------------------------------------------------------------------


    char error[1000];
    m = mj_loadXML("inverted_pendulum.xml", 0, error, 1000);
    d = mj_makeData(m);


    glfwInit();


    GLFWwindow* window = glfwCreateWindow(1200, 900, "DDPG", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);


    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);


    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);


    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);



    mjcb_control = test_controller;

    d->qpos[0] = 0.0;
    d->qpos[1] = 0.5 * 2 * (randu() - 0.5) * 15 * 2 * 3.1416 / 360;


    while (!glfwWindowShouldClose(window)) {

        mjtNum simstart = d->time;



        while (d->time - simstart < 1.0 / 60.0) {

            e = d->qpos[0];
            ie += e * m->opt.timestep;
            xd = 5.0*sin(d->time);

            mj_step(m, d);


        }

        /*if (d->time > 10) {
            ie = 0;
            mj_resetData(m, d);

            d->qpos[0] = 0.5*2 * (randu() - 0.5) * 3;
            d->qpos[1] = 0.5*2 * (randu() - 0.5) * 15 * 2 * 3.1416 / 360;
        }*/

        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);


        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);


        glfwSwapBuffers(window);


        glfwPollEvents();
     }


    mjv_freeScene(&scn);
    mjr_freeContext(&con);


    mj_deleteData(d);
    mj_deleteModel(m);


#if defined(__APPLE__)  defined(_WIN32)
    glfwTerminate();
#endif

    return 0;
}