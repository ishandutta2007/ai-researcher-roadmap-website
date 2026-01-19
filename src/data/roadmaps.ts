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
        title: 'Coming Soon',
        description: 'This roadmap is under construction.',
        resources: [],
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
