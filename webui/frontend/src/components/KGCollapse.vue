<template>
    <b-list-group-item>
        <b-button variant="outline-dark"
            @click="visible = !visible"
            >
            <b-icon-caret-down />
            {{property.name}}
        </b-button>
        <b-collapse v-model="visible">
            <div v-if="Array.isArray(property.values)">

                <div v-if="withId">
                    <b-list-group v-for="value in property.values" :key="value['@id']">
                        <b-list-group-item>
                            <k-g-link :to="value['@id']" />
                        </b-list-group-item>
                    </b-list-group>
                </div>
                <div v-else-if="withValue">
                    <b-list-group v-for="value in property.values" :key="value['@value']">
                        <b-list-group-item>
                            "{{value["@value"]}}"@<i>{{value["@language"]}}</i>
                        </b-list-group-item>
                    </b-list-group>

                </div>
                <div v-else>
                    <b-list-group v-for="value in property.values" :key="value">
                        <b-list-group-item>
                            <k-g-link :to="value" />
                        </b-list-group-item>
                    </b-list-group>
                </div>
           </div>
           <!-- No array too loop over -->
            <div v-else>
                {{strip(property.values)}}
            </div>
        </b-collapse>
    </b-list-group-item>
</template>

<script lang="ts">
import Vue from 'vue';
import { BIconCaretDown } from 'bootstrap-vue'
import KGLink from '@/components/KGLink.vue'
import {stripPrefix} from '../api'

export default Vue.extend({
  name: 'KGCollapse',

  components: {
      BIconCaretDown,
      KGLink
  },

  data() {
      return {
          visible: true
      }
  },

  computed: {
      withId() {
          return this.property.values && this.property.values[0]['@id']
      },

      withValue() {
          return this.property.values && this.property.values[0]['@value']
      }
  },

  props: ['property'],

  methods: {
      strip: stripPrefix
  },
  
});

</script>