import Component from "vue-class-component"

// Regtister the router hooks with their names
Component.registerHooks([
    "beforeRouteEnter",
    "beforeRouteLeave",
    "beforeRouteUpdate"
]);
