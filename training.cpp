#include "environment.h"
#include "Header.h"



using namespace arma;
using namespace mlpack;
using namespace std;
using namespace ens;


extern FFN<EmptyLoss, GaussianInitialization>* policy;



void train(void){

    mjcb_control = controller;

    // Set up the replay method.
    RandomReplay<ContinuousCartPole> replayMethod(30, 500);

    // Set up the training configuration.
    TrainingConfig config;
    config.StepSize() = 0.01;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;
    config.Discount() = 1.00;

    // Set up Actor network.
    static FFN<EmptyLoss, GaussianInitialization>  policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
    policy = &policyNetwork;
    policyNetwork.Add(new Linear(13));
    policyNetwork.Add(new ReLU());
    policyNetwork.Add(new Linear(1));
    policyNetwork.Add(new TanH());

    // Set up Critic network.
    FFN<EmptyLoss, GaussianInitialization>  qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
    qNetwork.Add(new Linear(9));
    qNetwork.Add(new ReLU());
    qNetwork.Add(new Linear(9));
    qNetwork.Add(new ReLU());
    qNetwork.Add(new Linear(1));

    // Set up the GaussianNoise parameters.
    int size = 1;
    double mu = 0.0;
    double sigma = 42.0; // 3.0;

    // Create an instance of the GaussianNoise class.
    GaussianNoise gaussianNoise(size, mu, sigma);

    // Set up Deep Deterministic Policy Gradient agent.
    DDPG<ContinuousCartPole, decltype(qNetwork), decltype(policyNetwork),
        GaussianNoise, AdamUpdate>
        agent(config, qNetwork, policyNetwork, gaussianNoise, replayMethod);

    std::vector<double> lastEpisodes;
    double R;
    double sum = 0.0;
    double maxAvg = 0.0;


    for (int i = 1; i < 4000; i++) {

        if (lastEpisodes.size() == 15) {
            lastEpisodes.erase(lastEpisodes.begin());
        }

        R = agent.Episode();

        lastEpisodes.push_back(R);



        sum = 0.0;

        for (int j = 0; j < 15; j++) {
            sum += lastEpisodes[j];
        }

        sum /= 15;

        cout << "Episode: " << i << ", Return: " << R << endl;

        if (i > 200) {
            maxAvg = (sum > maxAvg) ? sum : maxAvg;
        }

        if ((i > 200) && (lastEpisodes.size() == 15) && (sum > 397)) {
            break;
        }
    }

    cout << "Maximum Average Return: " << maxAvg;

    agent.Deterministic() = true;
    cout << "Test\n";
    cout << agent.Episode() << endl;


}