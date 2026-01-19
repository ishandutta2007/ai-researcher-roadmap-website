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
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
      },
    ],
  },
  'core-ml': {
    title: 'Core Machine Learning',
    description: 'Supervised, Unsupervised, and Reinforcement Learning',
    steps: [
      {
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
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
