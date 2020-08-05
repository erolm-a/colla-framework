<template>
    <router-link :to="getUrl">{{stripPrefix}}</router-link>
</template>

<script lang="ts">
import Vue from 'vue';

export default Vue.extend({
    name: 'KGLink',

    props: ["to"],

    computed: {
        getUrl: function() {
            const stripped =  this.stripPrefix;
            return "/kg/" + stripped;
        },

        stripPrefix: function() {
          const kglNamespace = "http://grill-lab.org/kg/entity/";
          const kglpropNamespace = "http://grill-lab.org/kg/property/"

          const to = this.to;

          if(to.startsWith(kglNamespace))
          {
            return "kgl:" + to.slice(kglNamespace.length);
          }
          else if(to.startsWith(kglpropNamespace))
          {
            return "kglprop:" + to.slice(kglpropNamespace.length);
          }

          return to;
      }
    },

    
});
</script>