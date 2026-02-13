import reactIMG from "./assets/react.png";
import componentsIMG from "./assets/components.png";
import jsxIMG from "./assets/jsx.png";
import stateIMG from "./assets/state.png";

export const CORE_CONCEPTS = [
    {
        title: "Components",
        description: "Building blocks of any React application. They allow you to split the UI into independent, reusable pieces.",
        img: componentsIMG
    },
    {
        title: "JSX",
        description: "A syntax extension for JavaScript that looks similar to XML or HTML. It allows you to write HTML-like code within JavaScript files.",
        img: jsxIMG
    },
    {
        title: "Props",
        description: "Short for 'properties', props are a way of passing data from parent to child components. They are read-only and help make components reusable.",
        img: reactIMG
    },
    {
        title: "State",
        description: "A built-in React object that allows components to create and manage their own data. State changes can trigger re-renders of the component.",
        img: stateIMG
    }
]