export type Roadmap = {
  title: string;
  description:string;
  steps: Step[];
};

export type Step = {
  title: string;
  description: string;
  resources: Resource[];
};

export type Resource = {
  title: string;
  url: string;
};

export const roadmaps: { [key: string]: Roadmap } = {
  'mathematics-for-ai': {
    title: 'Mathematics for AI',
    description: 'Linear Algebra, Calculus, Probability & Statistics',
    steps: [
      {
        title: 'Linear Algebra',
        description: 'Understand vectors, matrices, tensors, and their operations. Essential for understanding how neural networks process data.',
        resources: [
          {
            title: 'Khan Academy: Linear Algebra',
            url: 'https://www.khanacademy.org/math/linear-algebra',
          },
          {
            title: '3Blue1Brown: Essence of Linear Algebra',
            url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi',
          },
        ],
      },
      {
        title: 'Calculus',
        description: 'Grasp derivatives, gradients, and the chain rule. Crucial for understanding optimization algorithms like gradient descent and backpropagation.',
        resources: [
          {
            title: 'Khan Academy: Calculus',
            url: 'https://www.khanacademy.org/math/calculus-1',
          },
          {
            title: 'MIT OpenCourseWare: Single Variable Calculus',
            url: 'https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/',
          },
        ],
      },
      {
        title: 'Probability & Statistics',
        description: 'Learn probability distributions, hypothesis testing, statistical inference, and Bayesian statistics. Fundamental for data analysis and modeling uncertainty.',
        resources: [
          {
            title: 'Khan Academy: Statistics and Probability',
            url: 'https://www.khanacademy.org/math/statistics-probability',
          },
          {
            title: 'StatQuest with Josh Starmer',
            url: 'https://www.youtube.com/playlist?list=PLblh5JKN_IZx04KsFh-c2_wR8S42e2a-X',
          },
        ],
      },
    ],
  },
  'cs-fundamentals': {
    title: 'Computer Science Fundamentals',
    description: 'Data Structures, Algorithms, and Programming',
    steps: [
      {
        title: 'Programming Proficiency',
        description: 'Master Python for AI development. Familiarity with Git/GitHub for version control and basic command-line operations are essential.',
        resources: [
          {
            title: 'Python for Everybody (University of Michigan)',
            url: 'https://www.py4e.com/',
          },
          {
            title: 'Git Handbook',
            url: 'https://guides.github.com/introduction/git-handbook/',
          },
          {
            title: 'Linux Command Line Basics',
            url: 'https://ryanstutorials.net/linuxtutorial/',
          },
        ],
      },
      {
        title: 'Core Computer Science Concepts',
        description: 'Understand fundamental data structures (e.g., arrays, linked lists, trees, graphs) and algorithms (e.g., sorting, searching, dynamic programming), along with Big O notation for complexity analysis.',
        resources: [
          {
            title: 'GeeksforGeeks: Data Structures',
            url: 'https://www.geeksforgeeks.org/data-structures/',
          },
          {
            title: 'GeeksforGeeks: Algorithms',
            url: 'https://www.geeksforgeeks.org/fundamentals-of-algorithms/',
          },
          {
            title: 'Big O Notation Explained',
            url: 'https://www.freecodecamp.org/news/big-o-notation-demystified-a-simple-explanation-with-examples/',
          },
        ],
      },
      {
        title: 'Software Development Principles & System Design',
        description: 'Grasp general software development principles and basic system design concepts, sufficient to implement and run machine learning algorithms.',
        resources: [
          {
            title: 'Clean Code (Book by Robert C. Martin)',
            url: 'https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882',
          },
          {
            title: 'System Design Interview – An insider\'s guide',
            url: 'https://www.amazon.com/System-Design-Interview-insiders-guide/dp/B08CMF2CQF',
          },
        ],
      },
    ],
  },
  'core-ml': {
    title: 'Core Machine Learning',
    description: 'Supervised, Unsupervised, and Reinforcement Learning',
    steps: [
      {
        title: 'Types of Learning',
        description: 'Understand the fundamental paradigms of machine learning: Supervised, Unsupervised, and Reinforcement Learning.',
        resources: [
          {
            title: 'IBM: Supervised vs Unsupervised Learning',
            url: 'https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning',
          },
          {
            title: 'Google AI: Reinforcement Learning',
            url: 'https://ai.google/research/teams/brain/reinforcement-learning/',
          },
        ],
      },
      {
        title: 'Classical Machine Learning Algorithms',
        description: 'Familiarize yourself with widely used algorithms such as Linear/Logistic Regression, Decision Trees, Support Vector Machines (SVMs), K-Nearest Neighbors (K-NN), K-Means Clustering, and Ensemble Methods (e.g., Random Forests, Gradient Boosting).',
        resources: [
          {
            title: 'Scikit-learn documentation',
            url: 'https://scikit-learn.org/stable/documentation.html',
          },
          {
            title: 'Machine Learning Crash Course (Google)',
            url: 'https://developers.google.com/machine-learning/crash-course',
          },
        ],
      },
      {
        title: 'Evaluation Metrics & Concepts',
        description: 'Learn how to evaluate model performance using metrics like accuracy, precision, recall, F1-score, and AUC. Understand concepts like bias-variance trade-off, overfitting, underfitting, and regularization techniques.',
        resources: [
          {
            title: 'Confusion Matrix & Evaluation Metrics',
            url: 'https://towardsdatascience.com/confusion-matrix-and-evaluation-metrics-for-multi-class-classification-80865a363aeb',
          },
          {
            title: 'Bias-Variance Trade-off',
            url: 'https://www.ibm.com/cloud/learn/bias-variance-tradeoff',
          },
        ],
      },
    ],
  },
  'deep-learning': {
    title: 'Deep Learning',
    description: 'Neural Networks, CNNs, RNNs, and Transformers',
    steps: [
      {
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
      },
    ],
  },
  nlp: {
    title: 'Natural Language Processing (NLP)',
    description: 'From Text Preprocessing to Large Language Models',
    steps: [
      {
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
      },
    ],
  },
  'computer-vision': {
    title: 'Computer Vision',
    description: 'Image Processing, Object Detection, and Generation',
    steps: [
      {
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
      },
    ],
  },
};
