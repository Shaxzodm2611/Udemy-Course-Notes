# React Essentials

### Components

Components (`.jsx`) wrap javascript, html, and css together into one file. Helping keep front-end codebases relatively small.

- A component can be defined like any other JS function. Its return body must be some sort of mark-up. Furthermore, components must be capitalized

To insert a component into the main program body i.e: App() => {}. We can either insert the component like any other HTML tag, or you can have the component be self-closing. For example:

```jsx
function Body(){
    return (
        <p> Hi! </p>
    );
}

function App(){
    return (
        <title> Website </title>
        <Body/> //or <Body> </Body>
    );
}

export default App
```

### Component Tree

Since .jsx code is built with the React compiler, the code that is written is not the same code that is delivered to the end-user. Using `ReactDOM`, we are able to serve a .jsx file as an entry-point into a specific element. In the **starting-project** example, we have a index.html file that contains a component (id = "root") which we use as the entry-point for the `App.jsx` file with following ReactDOM methods in our index.jsx file:

```jsx
import App from "./App.jsx";
import ReactDOM from "react-dom/client";

const entryPoint = document.getElementById("root");
ReactDOM.createRoot(entryPoint).render(<App />);
```

Essentially, React compiles the **Component Tree** into the **DOM Tree**

### Outputting Dynamic Values in .JSX

- A single pair of curly braces tell the React compiler that in-between that _element_ or _attribute_ is a dynamic value. In between those curly braces, you can put any javascript statement **except** for loops, if statements, function definitions and block statements
- Say we have a name that is randomly picked from a collection of names. Instead of having all of that logic inside the mark-up block we can define a const within the main .jsx block and put the result of the .js function into that function. This is considered **good practice**

```jsx
const words = ["Jerry", "Ari", "Bob"];

function getRandomNum(max) {
  return Math.floor(Math.random() * (max + 1));
}
function Header() {
  return (
    <header>
      <p> Hi my name is {words[getRandomNum(3)]} </p>
    </header>
  );
}

/**********************************************
 ****************IS WORSE THAN******************
 ***********************************************/
function Header() {
  const name = words[getRandomNum(3)];
  return (
    <header>
      <p> Hi my name is {name} </p>
    </header>
  );
}
```

**Setting HTML attributes dynamically**

Say we have an image in a directory: `src/images/myImage.png`, instead doing it the typical HTML way: `<img src = "src/images/myImage.png>`, we can _import_ it as a variable. This has two benefits:

- Makes the markdown block cleaner
- Ensures that the image file that is being referenced does not get _lost_ at run-time/deployment

```jsx
//We have image at: src/images/myImage.png

import reactIMG from './src/images/myImage.png';

Return (
    <img src = {reactIMG} alt = "Some image">
);
```

_note: quotes should be omitted from the dynamically set attribute_

### Props

Props allow us to pass data into components and use that data within the component. **Prop** is simply the name used for components when "custom" attributes are passed to components. The custom attributes passed into the component/prop can be accessed via dot notation of the `prop` parameter that is passed into the component:

```jsx
function Header(props) {
  return <h1> Hello {props.name} </h1>;
}

function App() {
  return <Header name="User" />;
}
```

#### Alternative Prop Syntax

Imagine we have a prop defined as such:

```js
export const conceptArr = [
  {
    name = 'bob'
    age = 21
  },
  {
    name = 'sally'
    age = 22
  }
]
```

Instead of accessing each property of the object in the arrays (i.e the prop is _User_): `<User name = conceptArr[0].name,age = conceptArr[0].age>`, we can use the **spread** operator: `<...conceptArr[0]>`, we can do something similar in the html block the prop returns.

```jsx
function User(props){
  ...
  return (
    <h1> name: {prop.name}</h1>
    <p> age: {prop.age} </p>
  );
}
```

We can use **object destructuring** to destructure the properties of the incoming object into the params of the function:

```jsx
function User({name, age}){
  return(
  <h1> name: {name} </h1>
  <p> age: {age} </p>
  );
}
```

### Good Project Structure

It is best to separate components into different files, as-well keep style files close to their components
