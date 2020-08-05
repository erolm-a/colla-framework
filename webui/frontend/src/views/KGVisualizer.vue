<template>
    <div class="container" style="text-align: left">
        <b-breadcrumb :items="path">
        </b-breadcrumb>
        <b-list-group v-for="predicate in predicates" :key="predicate.name">
            <k-g-collapse :property="predicate" />
        </b-list-group> 
    </div>
</template>

<script lang="ts">
import Vue from 'vue';
import {searchKGItem} from '../api';
import KGCollapse from '@/components/KGCollapse.vue';

export default Vue.extend({
  name: 'KGVisualizer',
  components: {
      KGCollapse
  },
  data() {
      return {
          path: [
              {
                text: this.$route.params.entity,
                href: "#"
              }
          ],
          entity: {},
          predicates: []
      }
  },
  

    watch: {
        '$route.params.entity': async function(id) {
            await this.updatePredicates(id)
        }
    },

    created: async function() {
        await this.updatePredicates(this.$route.params.entity)
    },

    methods: {
        updatePredicates: async function(entityId) {
            try {
                console.log(this.$route.params.entity);
                const entity = (await searchKGItem(entityId))[0];
                const predicateObjects = Object.entries(entity);
                this.predicates = []
                for(const [predicate, nested] of predicateObjects)
                {
                    const newLeaf = {name: predicate, values: nested};
                    this.predicates.push(newLeaf);
                }
            }
            catch(error)
            {
                console.error(error);
            }
                            
        }
    }
});

</script>