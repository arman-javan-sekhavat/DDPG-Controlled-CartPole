#pragma once


#include <mlpack/prereqs.hpp>
#include <mlpack.hpp>
#include <mujoco/mujoco.h>

extern mjtNum force_scale = 7.0; //0.5;
mjtNum force = 0.0;

void controller(const mjModel* m, mjData* d) {
    d->ctrl[0] = force_scale*force;
}

using namespace arma;


double reward(double x) {
    if (fabs(x) < 1.5) {
        return 1.0 + 2.0 / cosh(x);
    }

    return 0.1;
}


namespace mlpack {

    /**
     * Implementation of Cart Pole task.
     */
    class ContinuousCartPole
    {
    public:
        /**
         * Implementation of the state of Cart Pole. Each state is a tuple vector
         * (position, velocity, angle, angular velocity).
         */
        class State
        {
        public:
            /**
             * Construct a state instance.
             */
            State() : data(dimension)
            { /* Nothing to do here. */
            }

            /**
             * Construct a state instance from given data.
             *
             * @param data Data for the position, velocity, angle and angular velocity.
             */
            State(const arma::colvec& data) : data(data)
            { /* Nothing to do here */
            }

            //! Modify the internal representation of the state.
            arma::colvec& Data() { return data; }

            //! Get the position.
            double Position() const { return data[0]; }
            //! Modify the position.
            double& Position() { return data[0]; }

            //! Get the velocity.
            double Velocity() const { return data[1]; }
            //! Modify the velocity.
            double& Velocity() { return data[1]; }

            //! Get the angle.
            double Angle() const { return data[2]; }
            //! Modify the angle.
            double& Angle() { return data[2]; }

            //! Get the angular velocity.
            double AngularVelocity() const { return data[3]; }
            //! Modify the angular velocity.
            double& AngularVelocity() { return data[3]; }

            //! Encode the state to a column vector.
            const arma::colvec& Encode() const { return data; }

            //! Dimension of the encoded state.
            static constexpr size_t dimension = 4;

        private:
            //! Locally-stored (position, velocity, angle, angular velocity).
            arma::colvec data;
        };

        /**
         * Implementation of action of Cart Pole.
         */
        class Action
        {
        public:
            // To store the action.

            Action() : action(1)
            { /* Nothing to do here */
            }

            std::vector<double> action;

            // Number of degrees of freedom.
            static const size_t size = 1;
        };

        /**
         * Construct a Cart Pole instance using the given constants.
         *
         * @param maxSteps The number of steps after which the episode
         *    terminates. If the value is 0, there is no limit.
         * @param tau The time interval.
         * @param thetaThresholdRadians The maximum angle.
         * @param xThreshold The maximum position.
         * @param doneReward Reward recieved by agent on success.
         */
        ContinuousCartPole(const size_t maxSteps = 0,
            double tau = 0.001,
            const double thetaThresholdRadians = 15 * 2 * 3.1416 / 360,
            const double xThreshold = 3.0,
            const double doneReward = 1.0
            ) :
            maxSteps(maxSteps),
            tau(tau),
            thetaThresholdRadians(thetaThresholdRadians),
            xThreshold(xThreshold),
            doneReward(doneReward),
            stepsPerformed(0)
        {
            char error[1000];
            m = mj_loadXML("inverted_pendulum.xml", 0, error, 1000);
            d = mj_makeData(m);
            tau = m->opt.timestep;
        }

        /**
         * Dynamics of Cart Pole instance. Get reward and next state based on current
         * state and current action.
         *
         * @param state The current state.
         * @param action The current action.
         * @param nextState The next state.
         * @return reward.
         */
        double Sample(const State& state,
            const Action& action,
            State& nextState)
        {
            // Update the number of steps performed.
            stepsPerformed++;

            // Calculate acceleration.
            e = d->qpos[0];
            ie += e*tau;

            force = action.action[0];

            // Update states.
            mj_step(m, d);

            //std::cout << d->qpos[1] << std::endl;

            nextState.Position() = d->qpos[0];
            nextState.Velocity() = d->qvel[0];
            nextState.Angle() = d->qpos[1];
            nextState.AngularVelocity() = d->qvel[1];

            

            // Check if the episode has terminated.
            bool done = IsTerminal(nextState);

            // Do not reward agent if it failed.
            //if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
            //    return -100;// doneReward;

            /**
             * When done is false, it means that the cartpole has fallen down.
             * For this case the reward is 1.0.
             */

            /*if (std::abs(state.Position()) > xThreshold ||
                std::abs(state.Angle()) > thetaThresholdRadians) {

                return -50;
            }*/

            return 2.0/cosh(d->qpos[0]);
        }

        /**
         * Dynamics of Cart Pole. Get reward based on current state and current
         * action.
         *
         * @param state The current state.
         * @param action The current action.
         * @return reward, it's always 1.0.
         */
        double Sample(const State& state, const Action& action)
        {
            State nextState;
            return Sample(state, action, nextState);
        }

        /**
         * Initial state representation is randomly generated within [-0.05, 0.05].
         *
         * @return Initial state for each episode.
         */
        State InitialSample()
        {
            stepsPerformed = 0;

            arma::colvec initialPose = arma::colvec(4, arma::fill::zeros);
            initialPose(0) = 2*(randu() - 0.5)*xThreshold*0.5;
            initialPose(2) = 2*(randu() - 0.5)*thetaThresholdRadians*0.5;

            m->qpos0[0] = initialPose(0);
            m->qpos0[2] = initialPose(2);

            return State(initialPose);

        }

        /**
         * This function checks if the cart has reached the terminal state.
         *
         * @param state The desired state.
         * @return true if state is a terminal state, otherwise false.
         */
        bool IsTerminal(const State& state)
        {
            if (maxSteps != 0 && stepsPerformed >= maxSteps)
            {
                Log::Info << "Episode terminated due to the maximum number of steps"
                    "being taken.";

                mj_resetData(m, d);
                ie = 0.0;
                return true;
            }
            else if (std::abs(state.Position()) > xThreshold ||
                std::abs(state.Angle()) > thetaThresholdRadians)
            {
                Log::Info << "Episode terminated due to agent failing.";
                mj_resetData(m, d);
                ie = 0.0;
                return true;
            }
            return false;
        }

        //! Get the number of steps performed.
        size_t StepsPerformed() const { return stepsPerformed; }

        //! Get the maximum number of steps allowed.
        size_t MaxSteps() const { return maxSteps; }
        //! Set the maximum number of steps allowed.
        size_t& MaxSteps() { return maxSteps; }

    private:

        double Kp = 0.1;
        double Ki = 0.1;
        double e = 0.0;
        double ie = 0.0;

        mjModel* m = NULL;
        mjData*  d = NULL;

        //! Locally-stored maximum number of steps.
        size_t maxSteps;

        //! Locally-stored time interval.
        double tau;

        //! Locally-stored maximum angle.
        double thetaThresholdRadians;

        //! Locally-stored maximum position.
        double xThreshold;

        //! Locally-stored done reward.
        double doneReward;

        //! Locally-stored number of steps performed.
        size_t stepsPerformed;

    };

} // namespace mlpack