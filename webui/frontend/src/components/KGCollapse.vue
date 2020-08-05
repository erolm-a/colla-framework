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

                <b-list-group v-for="value in property.values" :key="value">
                    <b-list-group-item>
                        <span v-if="value['@id']">
                            <k-g-link :to="value['@id']" />
                        </span>

                        <span v-else-if="value['@value']">
                            "{{value["@value"]}}"@<i>{{value["@language"]}}</i>
                        </span>

                        <span v-else>
                            <k-g-link :to="value" />
                        </span>
                    </b-list-group-item>
                </b-list-group>
            </div>
            <div v-else>
                {{stripPrefix(property.values)}}
            </div>
        </b-collapse>
    </b-list-group-item>
</template>

<script lang="ts">
import Vue from 'vue';
import {searchKGItem} from '../api';
import { BIconCaretDown } from 'bootstrap-vue'
import KGLink from '@/components/KGLink.vue'

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

  props: ['property'],

});

</script>