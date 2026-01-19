export type Roadmap = {
  title: string;
  description: string;
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
  frontend: {
    title: "Frontend Developer",
    description: "Step by step guide to becoming a modern frontend developer in 2024",
    steps: [
      {
        title: "Internet",
        description: "How does the internet work? What is HTTP? How do browsers work?",
        resources: [
          {
            title: "How the Internet Works",
            url: "https://developer.mozilla.org/en-US/docs/Learn/Common_questions/How_does_the_Internet_work",
          },
        ],
      },
      {
        title: "HTML",
        description: "Learn the basics of HTML, the standard markup language for creating web pages.",
        resources: [
          {
            title: "HTML Tutorial",
            url: "https://www.w3schools.com/html/",
          },
        ],
      },
      {
        title: "CSS",
        description: "Learn the basics of CSS, a style sheet language used for describing the presentation of a document written in a markup language like HTML.",
        resources: [
          {
            title: "CSS Tutorial",
            url: "https://www.w3schools.com/css/",
          },
        ],
      },
      {
        title: "JavaScript",
        description: "Learn the basics of JavaScript, a programming language that enables you to create dynamically updating content, control multimedia, animate images, and pretty much everything else.",
        resources: [
          {
            title: "JavaScript Tutorial",
            url: "https://www.w3schools.com/js/",
          },
        ],
      },
    ],
  },
};
