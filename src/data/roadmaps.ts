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
        title: 'Neural Network Basics',
        description: 'Understand the building blocks of neural networks: perceptrons, multi-layer perceptrons (MLPs), activation functions (ReLU, Sigmoid, Tanh), loss functions, backpropagation, and optimizers (SGD, Adam, RMSprop).',
        resources: [
          {
            title: 'Deep Learning (Goodfellow, Bengio, Courville) - Chapter 6',
            url: 'https://www.deeplearningbook.org/contents/mlp.html',
          },
          {
            title: '3Blue1Brown: Neural Networks series',
            url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi',
          },
        ],
      },
      {
        title: 'Foundational Architectures',
        description: 'Explore key deep learning architectures: Convolutional Neural Networks (CNNs) for computer vision, Recurrent Neural Networks (RNNs) including LSTMs and GRUs for sequential data, and Feedforward Deep Neural Networks (DNNs).',
        resources: [
          {
            title: 'CS231n: Convolutional Neural Networks for Visual Recognition',
            url: 'http://cs231n.stanford.edu/',
          },
          {
            title: 'The Unreasonable Effectiveness of Recurrent Neural Networks',
            url: 'http://karpathy.github.io/2015/05/21/rnn-effectiveness/',
          },
        ],
      },
      {
        title: 'Advanced Architectures',
        description: 'Delve into advanced architectures like Transformers and Attention Mechanisms (pivotal for NLP) and Generative Models such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).',
        resources: [
          {
            title: 'Attention Is All You Need (Transformer paper)',
            url: 'https://arxiv.org/abs/1706.03762',
          },
          {
            title: 'Generative Adversarial Networks (GANs) Explained',
            url: 'https://developers.google.com/machine-learning/gan/gan_structure',
          },
        ],
      },
    ],
  },
  nlp: {
    title: 'Natural Language Processing (NLP)',
    description: 'From Text Preprocessing to Large Language Models',
    steps: [
      {
        title: 'Text Preprocessing',
        description: 'Learn techniques like tokenization, lemmatization, stemming, stopword removal, and text cleaning. Utilize libraries like NLTK and SpaCy.',
        resources: [
          {
            title: 'NLTK Book',
            url: 'https://www.nltk.org/book/',
          },
          {
            title: 'SpaCy 101: What\'s New',
            url: 'https://spacy.io/usage/spacy-101',
          },
        ],
      },
      {
        title: 'Text Representation and Feature Extraction',
        description: 'Understand traditional methods like Bag of Words (BoW) and TF-IDF, as well as modern word embeddings (Word2Vec, GloVe, FastText) and document embeddings (Doc2Vec).',
        resources: [
          {
            title: 'A Gentle Introduction to Word2Vec',
            url: 'https://machinelearningmastery.com/gentle-introduction-word-embeddings/',
          },
          {
            title: 'Understand TF-IDF',
            url: 'https://www.datacamp.com/community/tutorials/tf-idf-tutorial',
          },
        ],
      },
      {
        title: 'Core NLP Tasks',
        description: 'Gain experience with tasks such as Sentiment Analysis, Named Entity Recognition (NER), Part-of-Speech (POS) Tagging, and Text Classification.',
        resources: [
          {
            title: 'Text Classification with NLTK and Scikit-learn',
            url: 'https://realpython.com/sentiment-analysis-python/',
          },
          {
            title: 'Named Entity Recognition (NER) with SpaCy',
            url: 'https://towardsdatascience.com/named-entity-recognition-ner-with-spacy-and-nltk-200a40ac2c8d',
          },
        ],
      },
      {
        title: 'Sequence Models',
        description: 'Study Recurrent Neural Networks (RNNs), including LSTMs and GRUs, and Sequence-to-Sequence Models with Encoder-Decoder architectures for handling sequential data.',
        resources: [
          {
            title: 'Illustrated Guide to LSTMs and GRUs',
            url: 'https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-4467c9d06b21',
          },
          {
            title: 'Sequence to Sequence Learning with Neural Networks (Paper)',
            url: 'https://arxiv.org/abs/1409.3215',
          },
        ],
      },
      {
        title: 'Attention Mechanisms and Transformers',
        description: 'Delve into the Transformer architecture, attention mechanisms, and work with pre-trained models like BERT and GPT.',
        resources: [
          {
            title: 'The Illustrated Transformer',
            url: 'http://jalammar.github.io/illustrated-transformer/',
          },
          {
            title: 'Hugging Face Transformers Course',
            url: 'https://huggingface.co/course/chapter1/1',
          },
        ],
      },
      {
        title: 'Large Language Models (LLMs) and Generative AI',
        description: 'Understand how LLMs are trained, and develop skills in Prompt Engineering and Retrieval-Augmented Generation (RAG).',
        resources: [
          {
            title: 'Stanford CS324: Large Language Models',
            url: 'https://stanford-cs324.github.io/winter2022/lectures/',
          },
          {
            title: 'Prompt Engineering Guide',
            url: 'https://www.promptingguide.ai/',
          },
        ],
      },
    ],
  },
  'computer-vision': {
    title: 'Computer Vision',
    description: 'Image Processing, Object Detection, and Generation',
    steps: [
      {
        title: 'Digital Image Fundamentals',
        description: 'Understand how images are represented (pixels, channels), image acquisition processes, and basic manipulation techniques like resizing and cropping.',
        resources: [
          {
            title: 'OpenCV-Python Tutorials: Core Operations',
            url: 'https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html',
          },
          {
            title: 'Digital Image Processing (Book by Gonzalez & Woods)',
            url: 'https://www.amazon.com/Digital-Image-Processing-Rafael-Gonzalez/dp/0133356720',
          },
        ],
      },
      {
        title: 'Image Processing & Traditional Methods',
        description: 'Explore techniques for image enhancement (noise reduction, sharpening), filtering (convolutions), and feature extraction (edge detection like Canny, corner detection like Harris). Gain familiarity with OpenCV.',
        resources: [
          {
            title: 'OpenCV-Python Tutorials: Image Processing',
            url: 'https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html',
          },
          {
            title: 'Introduction to Computer Vision (Course)',
            url: 'https://www.coursera.org/learn/introduction-to-computer-vision',
          },
        ],
      },
      {
        title: 'Deep Learning for Computer Vision',
        description: 'Master Convolutional Neural Networks (CNNs) and their architectures (e.g., ResNet, VGG). Learn about key tasks like Image Classification, Object Detection (YOLO, SSD, R-CNNs), Image Segmentation, and Transfer Learning.',
        resources: [
          {
            title: 'CS231n: Convolutional Neural Networks for Visual Recognition',
            url: 'http://cs231n.stanford.edu/',
          },
          {
            title: 'YOLO (You Only Look Once) Explained',
            url: 'https://towardsdatascience.com/yolo-you-only-look-once-explained-and-implemented-with-keras-and-tensorflow-d3a95d732849',
          },
        ],
      },
      {
        title: 'Advanced Computer Vision Topics',
        description: 'Delve into cutting-edge areas such as Transformers for Vision (ViT), Generative Adversarial Networks (GANs) for image generation, 3D Computer Vision (3D Reconstruction, NeRFs), and Video Understanding.',
        resources: [
          {
            title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT paper)',
            url: 'https://arxiv.org/abs/2010.11929',
          },
          {
            title: 'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis',
            url: 'https://www.matthewtancik.com/nerf',
          },
        ],
      },
    ],
  },
};
